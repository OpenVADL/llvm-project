//===- ScopMatcherPatternEmitter.cpp - Emitter for ScopMatcherPatterns ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>
#include <sstream>

#define DEBUG_TYPE "scop-matcher-pattern-emitter"

using namespace llvm;

namespace PermSeq {

class node;
using vvn = std::vector<std::vector<node>>;
using vn = std::vector<node>;

vvn perms(vn v);
vvn combine(vvn ls, vvn rs);

enum kind {
  S, // Sequence
  P, // Permutation
  O, // Or
  V  // Value
};

class node {
public:
  node(vn cs, kind k, char u = 'x') : cs(cs), k(k), v(u) {
    if (k == S) v = 's';
    else if (k == P) v = 'p';
    else if (k == O) v = 'o';
  }

  node() : cs({}), k(V), v('d') {}

  vn cs;
  kind k;
  char v;

  void replace_perms() {
    if (k == P) {
      vvn ps = perms(cs);
      cs.clear();
      k = O;
      v = 'o';
      for (vn p: ps) {
        node new_s(p, S);
        cs.push_back(new_s);
      }
    }
    for (node &c: cs) {
      c.replace_perms();
    }
  }

  vvn collect() {
    vvn r;
    if (k == O) {
      for (node c: cs) {
        vvn us = c.collect();
        for (vn u: us) {
          r.push_back(u);
        }
      }
    }
    else if (k == S) {
      for (node c: cs) {
        if (r.empty()) {
          r = c.collect();
        }
        else {
          vvn t = r;
          r.clear();
          vvn ct = c.collect();
          r = combine(t, ct);
        }
      }
    }
    else if (k == V) {
      vn f;
      f.push_back(*this);
      r.push_back(f);
    }
    return r;
  }
};

vvn perms(vn v) {
  vvn r;
  if (v.size() == 1) {
    vn s;
    s.push_back(v[0]);
    r.push_back(s);
    return r;
  }

  for (size_t i = 0; i < v.size(); ++i) {
    node n = v[i];
    vn c = v;
    c.erase(c.begin() + i);
    vvn t = perms(c);
    for (vn tv: t) {
      tv.push_back(n);
      r.push_back(tv);
    }
  }
  return r;
}

vvn combine(vvn ls, vvn rs) {
  vvn p;
  for (vn l: ls) {
    for (vn r: rs) {
      vn d = l;
      d.insert(d.end(), r.begin(), r.end());
      p.push_back(d);
    }
  }
  return p;
}

} // namespace PermutationSequence

namespace {
class ScopMatcherPatternEmitter {
  RecordKeeper &Records;

public:
  ScopMatcherPatternEmitter (RecordKeeper &R) : Records(R) {}
  void run(raw_ostream &OS);
};

class Traversal {
public:
  RecordKeeper &RK;

  Traversal (RecordKeeper &RK) : RK(RK) {}

  virtual void handleStore(Record *n, Record *val, Record *addr) { trav(val); trav(addr); };
  virtual void handleLoad (Record *n, Record *addr) { trav(addr); };
  virtual void handleFAdd (Record *n, Record *l, Record *r) { trav(l); trav(r); };
  virtual void handleFMul (Record *n, Record *l, Record *r) { trav(l); trav(r); };
  virtual void handlePlaceholder (Record *n) {};

  void trav(Record *p) {
    bool isIPat        = false;
    bool isStore       = false;
    bool isLoad        = false;
    bool isFMul        = false;
    bool isFAdd        = false;
    bool isPlaceholder = false;

    ArrayRef<std::pair<Record *, SMRange>> SCs = p->getSuperClasses();
    for (auto scp: SCs) {
      Record *sc = scp.first;
      isIPat        |= RK.getClass("IPat") == sc;
      isLoad        |= RK.getClass("Load") == sc;
      isStore       |= RK.getClass("Store") == sc;
      isFAdd        |= RK.getClass("FAdd") == sc;
      isFMul        |= RK.getClass("FMul") == sc;
      isPlaceholder |= RK.getClass("Placeholder") == sc;
    }
    if (isStore) {
      Record *v = p->getValueAsDef("val");
      Record *a = p->getValueAsDef("addr");
      handleStore(p, v, a);
    }
    else if (isLoad) {
      Record *a = p->getValueAsDef("addr");
      handleLoad(p, a);
    }
    else if (isFAdd) {
      Record *l = p->getValueAsDef("l");
      Record *r = p->getValueAsDef("r");
      handleFAdd(p, l, r);
    }
    else if (isFMul) {
      Record *l = p->getValueAsDef("l");
      Record *r = p->getValueAsDef("r");
      handleFMul(p, l, r);
    }
    else if (isPlaceholder) {
      handlePlaceholder(p);
    }
    else
      llvm_unreachable("Unknown node type");
  }
};

class CountNodes : public Traversal {
public:
  int N;
  CountNodes (RecordKeeper &RK) : Traversal(RK), N(0) {}
  virtual void handleStore(Record *n, Record *val, Record *addr ) override { N++; trav(val); trav(addr); };
  virtual void handleLoad (Record *n, Record *addr) override { N++; trav(addr); };
  virtual void handleFAdd (Record *n, Record *l, Record *r) override { N++; trav(l); trav(r); };
  virtual void handleFMul (Record *n, Record *l, Record *r) override { N++; trav(l); trav(r); };
};

class CountCommutative : public Traversal {
public:
  int N;
  CountCommutative(RecordKeeper &RK) : Traversal(RK), N(0) {}
  virtual void handleFAdd (Record *n, Record *l, Record *r) override { N++; trav(l); trav(r); };
  virtual void handleFMul (Record *n, Record *l, Record *r) override { N++; trav(l); trav(r); };
};

class CountLoadsStores: public Traversal {
public:
  llvm::StringMap<int> &loads;
  llvm::StringMap<int> &stores;
  CountLoadsStores (RecordKeeper &RK, llvm::StringMap<int> &loads, llvm::StringMap<int> &stores)
    : Traversal(RK), loads(loads), stores(stores) {}

  virtual void handleLoad (Record *n, Record *addr) override {
    llvm::StringRef s = addr->getName();
    loads[s]++;
  };
  virtual void handleStore (Record *n, Record *v,  Record *addr) override {
    llvm::StringRef s = addr->getName();
    stores[s]++;
    trav(v);
  };
};

class CountPlaceholders: public Traversal {
public:
  llvm::StringMap<int> &phs;
  CountPlaceholders (RecordKeeper &RK, llvm::StringMap<int> &phs)
    : Traversal(RK), phs(phs) {}
  virtual void handlePlaceholder (Record *n) override {
    llvm::StringRef s = n->getName();
    phs[s]++;
  };
};

class PrintNodes : public Traversal {
public:
  raw_ostream &OS;
  unsigned commBits;
  PrintNodes (RecordKeeper &RK, raw_ostream &OS, unsigned commBits)
    : Traversal(RK), OS(OS), commBits(commBits) {}
  virtual void handleStore(Record *n, Record *val, Record *addr ) override {
    OS << "      m_Store(\n";
    trav(val);
    OS << "      , m_Value(" << addr->getName() << ")\n";
    OS << "      , S" << addr->getName() << ")\n";
  };
  virtual void handleLoad (Record *n, Record *addr) override {
    llvm::StringRef s = addr->getName();
    OS << "      m_Load(\n";
    OS << "      m_Value(" << s << ")\n";
    OS << "      , L" << s << ")\n";
  };
  virtual void handleFAdd (Record *n, Record *l, Record *r) override {
    bool swap = commBits & 1;
    commBits >>= 1;
    OS << "      m_FAdd(\n";
    if (swap) {
      trav(r);
      OS << "      ,\n";
      trav(l);
    }
    else {
      trav(l);
      OS << "      ,\n";
      trav(r);
    }
    OS << "      )\n";
  };
  virtual void handleFMul (Record *n, Record *l, Record *r) override {
    bool swap = commBits & 1;
    commBits >>= 1;
    OS << "      m_FMul(\n";
    if (swap) {
      trav(r);
      OS << "      ,\n";
      trav(l);
    }
    else {
      trav(l);
      OS << "      ,\n";
      trav(r);
    }
    OS << "      )\n";
  }
};

PermSeq::node permSeqFromInterchange(Record *p, std::vector<char> &dimIDstack) {
  if (p->isSubClassOf("Leaf")) {
    if (dimIDstack.empty())
      llvm_unreachable("Interchange definition uses too many dimensions");

    char id = dimIDstack.back();
    dimIDstack.pop_back();
    PermSeq::node r({}, PermSeq::V, id);
    return r;
  }
  else {
    std::vector<Record *> children = p->getValueAsListOfDefs("i");
    PermSeq::vn ncs;
    for (Record *c: children) {
      PermSeq::node nc = permSeqFromInterchange(c, dimIDstack);
      ncs.push_back(nc);
    }
    if (p->isSubClassOf("Seq")) {
      PermSeq::node r(ncs, PermSeq::S);
      return r;
    }
    else if (p->isSubClassOf("Perm")) {
      PermSeq::node r(ncs, PermSeq::P);
      return r;
    }
    else
      llvm_unreachable("Must be Leaf, Seq or Perm");
  }
}

std::vector<std::string> makeDimensionStrings(PermSeq::vvn interchangeList) {
  std::vector<std::string> r;
  for (PermSeq::vn ns: interchangeList) {
    std::stringstream ss;
    for (PermSeq::node n: ns) {
      ss << n.v << ",";
    }
    std::string l = ss.str();
    l.pop_back();
    r.push_back(l);
  }
  return r;
}

} // namespace

void ScopMatcherPatternEmitter::run(raw_ostream &OS) {
  for (Record *R : Records.getAllDerivedDefinitions("ScopMatcherPattern")) {

    llvm::StringRef className = R->getName();

    // Dimensions
    Record *m = R->getValueAsDef("mode");
    bool isParameterized = false;
    bool isFixed         = false;

    ArrayRef<std::pair<Record *, SMRange>> SCs = m->getSuperClasses();
    for (auto scp: SCs) {
      Record *sc = scp.first;
      isParameterized |= Records.getClass("Parameterized") == sc;
      isFixed         |= Records.getClass("Fixed") == sc;
    }

    std::vector<int64_t> dimensions;
    if (isFixed) {
      dimensions = m->getValueAsListOfInts("fixedDims");
    }
    else if (isParameterized) {
      int numDims = m->getValueAsInt("numDims");
      std::vector<int64_t> d(numDims, 0);
      dimensions = d;
    }
    else
      llvm_unreachable("Must be either Fixed or Parameterized");

    if (dimensions.size() < 1 || dimensions.size() > 4)
      llvm_unreachable("Pattern generator supports 1 to 4 dimensions");

    // Instruction pattern
    Record *p = R->getValueAsDef("iPat");

    CountNodes cn(Records);
    cn.trav(p);
    int nodeCount = cn.N;

    CountCommutative cc(Records);
    cc.trav(p);
    unsigned commutativeCount = cc.N;

    llvm::StringMap<int> loads;
    llvm::StringMap<int> stores;
    CountLoadsStores cl(Records, loads, stores);
    cl.trav(p);

    llvm::StringMap<int> phs;
    CountPlaceholders cp(Records, phs);
    cp.trav(p);

    // Access pattern
    std::vector<Record *> accesses = R->getValueAsListOfDefs("accesses");
    // Where do we need strides
    llvm::StringMap<char> strides;
    for (Record *a: accesses) {
      Record *ph = a->getValueAsDef("placeholder");
      std::vector<Record *> ds = a->getValueAsListOfDefs("dims");
      int dsize = ds.size();
      if (dsize == 2) {
        if (a->isSubClassOf("ReadAccess")) {
          strides[ph->getName()] = 'L';
        }
        if (a->isSubClassOf("WriteAccess") && ! strides.contains(ph->getName())) {
          strides[ph->getName()] = 'S';
        }
      }
      if (dsize > 2)
        llvm_unreachable("Pattern generator supports a max of one stride per array");
    }

    // Emitter
    Record *em = R->getValueAsDef("emitter");
    StringRef func = em->getValueAsString("function");
    std::vector<Record *> args = em->getValueAsListOfDefs("args");

    llvm::Twine us("_");

    // At the moment these are the only dimensions avaiable in the
    // tablegen file.
    std::vector<char> dimensionIDstack;
    dimensionIDstack.push_back('n');
    dimensionIDstack.push_back('m');
    dimensionIDstack.push_back('l');
    dimensionIDstack.push_back('k');
    dimensionIDstack.push_back('j');

    // Interchange
    Record *ic = R->getValueAsDef("interchange");
    PermSeq::node interchange = permSeqFromInterchange(ic, dimensionIDstack);
    interchange.replace_perms();
    PermSeq::vvn interchangeList = interchange.collect();

    std::vector<std::string> dimensionStrings = makeDimensionStrings(interchangeList);

    OS << "#if 0\n";
    OS << "#endif\n";

    for (int interchangeCount = 0; interchangeCount < dimensionStrings.size(); ++interchangeCount) {
      for (unsigned commutation = 0; commutation < (1u << commutativeCount); commutation++) {
        PrintNodes nodePrinter(Records, OS, commutation);

        llvm::Twine commutationString(commutation);
        llvm::Twine interchnageCountString(interchangeCount);
        llvm::Twine extendedClassName(className + us + commutationString + us + interchnageCountString);

        OS <<                            "class " << extendedClassName << " : public StmtPattern {\n";
        OS <<                            "public:\n";
        OS <<                            "  " << extendedClassName << "(const Dependences &D) : StmtPattern(D) {}\n";
        OS <<                            "\n";
        for (auto k: loads.keys()) {
          OS <<                          "  Instruction *L" << k << " = nullptr;\n";
        }
        for (auto k: stores.keys()) {
          OS <<                          "  Instruction *S" << k << " = nullptr;\n";
        }
        OS <<                            "\n";
        for (auto k: strides.keys()) {
          OS <<                          "  Value *stride" << k << " = nullptr;\n";
        }
        OS <<                            "\n";
        OS <<                            "  unsigned int getNodeCount() override { return " << nodeCount << "; }\n";
        OS <<                            "\n";
        OS <<                            "  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {\n";

        OS <<                            "    return new " << func << "(Stmt";
        for (Record *arg: args) {
          OS << ", ";
          if (loads.contains(arg->getName())) {
            OS << "L" << arg->getName();
          }
          else if (stores.contains(arg->getName())) {
            OS << "S" << arg->getName();
          }
          else if (arg->isSubClassOf("Empty")) {
            OS << "nullptr";
          }
          else {
            OS << arg->getName();
          }
        }
        OS << ");\n";
        OS <<                            "  }\n";
        OS <<                            "\n";
        OS <<                            "  virtual std::vector<unsigned> getDimensions() override { return {";
        for (int i: dimensions) {
          OS <<                                                                                       i << ",";
        }
        OS <<                            "}; }\n";
        OS <<                            "\n";
        OS <<                            "  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {\n";
        for (auto k: loads.keys()) {
          OS <<                          "    Value *" << k << ";\n";
        }
        OS <<                            "    auto P = \n";
        nodePrinter.trav(p);
        OS <<                            "    ;\n";

        OS <<                            "\n";
        OS <<                            "    bool r = PatternMatch::match(I, P);\n";
        OS <<                            "    if (r) {\n";
        OS <<                            "      std::map<Instruction *, const char *> rv;\n";
        for (Record *a: accesses) {
          OS <<                          "      rv[";
          Record *ph = a->getValueAsDef("placeholder");
          if (a->isSubClassOf("ReadAccess")) {
            OS <<                                 "L";
          }
          else if (a->isSubClassOf("WriteAccess")) {
            OS <<                                 "S";
          }
          else
            llvm_unreachable("Access must be read or write");
          OS <<                                    ph->getName() << "] = \"{ S[";
          OS <<                                    dimensionStrings[interchangeCount];
          OS <<                                    "] -> " << ph->getName() << "[";
          std::vector<Record *> ds = a->getValueAsListOfDefs("dims");
          int dsize = ds.size();
          for (int i = 0; i < dsize - 1; i++) {
            OS <<                                  ds[i]->getName() << ", ";
          }
          OS <<                                    ds[dsize-1]->getName() << "] }\";\n";
        }
        OS <<                            "      return rv;\n";
        OS <<                            "    }\n";
        OS <<                            "    else {\n";
        OS <<                            "      return std::nullopt;\n";
        OS <<                            "    }\n";
        OS <<                            "  }\n";
        OS <<                            "\n";
        OS <<                            "  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) override {\n";
        OS <<                            "    if ( ";
        for (auto k: strides.keys()) {
          OS <<                          "I != " << strides[k] << k << " && ";
        }
        OS <<                            "true )\n";
        OS <<                            "      return;\n";
        OS <<                            "\n";
        OS <<                            "    // FIXME Is there a way to ensure the dimensionIndex is adaptable and correct?\n";
        OS <<                            "    unsigned dimensionIndex = 1;\n";
        OS <<                            "\n";
        OS <<                            "    const SCEV *sc = SAI->getDimensionSize(dimensionIndex);\n";
        OS <<                            "    SetVector<Value *> values;\n";
        OS <<                            "    findValues(sc, *SE, values);\n";
        OS <<                            "\n";
        OS <<                            "    if (values.size() > 0) {\n";
        OS <<                            "      Value *w = values[0];\n";
        for (auto k: strides.keys()) {
          OS <<                          "      if (I == " << strides[k] << k << ") {\n";
          OS <<                          "        stride" << k << " = w;\n";
          OS <<                          "        LLVM_DEBUG(dbgs() << \"stride" << k << " is \" << *stride" << k << " << \"\\n\");\n";
          OS <<                          "      }\n";
        }
        OS <<                            "      return;\n";
        OS <<                            "    }\n";
        OS <<                            "\n";
        OS <<                            "    if (sc->getSCEVType() == SCEVTypes::scConstant) {\n";
        OS <<                            "      ConstantInt *intVal = static_cast<const SCEVConstant *>(sc)->getValue();\n";
        OS <<                            "\n";
        OS <<                            "      // Ugly type conversion\n";
        OS <<                            "      Module *M = Stmt.getParent()->getFunction().getParent();\n";
        OS <<                            "      Type *intType = Type::getInt64Ty(M->getContext());\n";
        OS <<                            "      Value *s = ConstantInt::getSigned(intType, intVal->getSExtValue());\n";
        for (auto k: strides.keys()) {
          OS <<                          "      if (I == " << strides[k] << k << ") {\n";
          OS <<                          "        stride" << k << " = s;\n";
          OS <<                          "        LLVM_DEBUG(dbgs() << \"stride" << k << " is \" << *stride" << k << " << \"\\n\");\n";
          OS <<                          "      }\n";
        }
        OS <<                            "      return;\n";
        OS <<                            "    }\n";
        OS <<                            "\n";
        OS <<                            "    llvm_unreachable(\"Cannot determine array sizes\");\n";
        OS <<                            "  }\n";
        OS <<                            "\n";
        OS <<                            "  virtual bool enableTiling() override { return ";
        if (isFixed)
          OS << "true";
        else
          OS << "false";
        OS <<                            ";  }\n";
        OS <<                            "};\n";
        OS <<                            "\n";
      }
    }
  }
}

static TableGen::Emitter::OptClass<ScopMatcherPatternEmitter>
    X("gen-scop-matcher-patterns", "Generate patterns for ScopMatcher");

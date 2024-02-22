//===-- ScopMatcher.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polly/ScopMatcher.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/SCEVValidator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "isl/map.h"
#include "isl/set.h"
#include <memory>
#include <string>
#include <sstream>
#include <system_error>

namespace cl = llvm::cl;

using namespace llvm;
using namespace polly;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "polly-scop-matcher"

#define SL 64
#define SM 64
#define SN 64

#define AMX_L 16
#define AMX_M 16
#define AMX_N 16

#define SME_L 8
#define SME_M 8
#define SME_N 8

#define INTRIWIDTH 4

namespace polly {

enum ScopMatcherChoice {
  SCOP_MATCHER_NONE,
  SCOP_MATCHER_EXP,
  SCOP_MATCHER_EXP_N,
  SCOP_MATCHER_DOTPROD,
  SCOP_MATCHER_DOTPROD_N,
  SCOP_MATCHER_DOTPROD_INTRINSIC,
  SCOP_MATCHER_AXPY,
  SCOP_MATCHER_AXPY_N,
  SCOP_MATCHER_VECT_MATMUL,
  SCOP_MATCHER_VECT_MATMUL_MN,
  SCOP_MATCHER_MATMUL,
  SCOP_MATCHER_MATMUL_LMN,
  SCOP_MATCHER_MATMUL_ALPHA_LMN,
  SCOP_MATCHER_MATMUL_AMX,
  SCOP_MATCHER_MATMUL_SME,
};

static cl::opt<ScopMatcherChoice> ScopMatcherSelected(
    "polly-select-scop-matcher", cl::desc("Which ScopMatcher should be used"),
    cl::values(clEnumValN(SCOP_MATCHER_NONE,    "none",    "Bypass ScopMatcher"),
               clEnumValN(SCOP_MATCHER_EXP, "exp", "Experimental ScopMatcher"),
               clEnumValN(SCOP_MATCHER_EXP_N, "exp-n", "Experimental N ScopMatcher"),
               clEnumValN(SCOP_MATCHER_DOTPROD, "dotprod", "DotProd ScopMatcher"),
               clEnumValN(SCOP_MATCHER_DOTPROD_N, "dotprod-n", "DotProd-N ScopMatcher"),
               clEnumValN(SCOP_MATCHER_DOTPROD_INTRINSIC, "dotprod-intr", "DotProd Intrinsic ScopMatcher"),
               clEnumValN(SCOP_MATCHER_AXPY, "axpy", "Axpy ScopMatcher"),
               clEnumValN(SCOP_MATCHER_AXPY_N, "axpy-n", "Axpy-N ScopMatcher"),
               clEnumValN(SCOP_MATCHER_VECT_MATMUL,  "vect-matmul",  "Vect-MatMul ScopMatcher"),
               clEnumValN(SCOP_MATCHER_VECT_MATMUL_MN,  "vect-matmul-mn",  "Vect-MatMul-mn ScopMatcher"),
               clEnumValN(SCOP_MATCHER_MATMUL,  "matmul",  "MatMul ScopMatcher"),
               clEnumValN(SCOP_MATCHER_MATMUL_LMN,  "matmul-lmn",  "MatMul-LMN ScopMatcher"),
               clEnumValN(SCOP_MATCHER_MATMUL_ALPHA_LMN,  "matmul-alpha-lmn",  "MatMul-Alpha-LMN ScopMatcher"),
               clEnumValN(SCOP_MATCHER_MATMUL_AMX,  "matmul-amx",  "MatMul-AMX ScopMatcher"),
               clEnumValN(SCOP_MATCHER_MATMUL_SME,  "matmul-sme",  "MatMul-SME ScopMatcher")
               ),
    cl::Hidden, cl::init(SCOP_MATCHER_NONE), cl::cat(PollyCategory));

class DotProdEmitter : public ReplacementEmitter {
  public:

  Instruction *LC;
  Instruction *LA;
  Instruction *LB;

  DotProdEmitter(ScopStmt &Stmt,
                 Instruction *LC,
                 Instruction *LA,
                 Instruction *LB)
    : ReplacementEmitter(Stmt), LC(LC), LA(LA), LB(LB) {}

  void emit(PollyIRBuilder &Builder,
            const DebugLoc &Loc,
            GenerateLocation genLoc) override {
    Module *M = Stmt.getParent()->getFunction().getParent();
    Type *voidType  = Type::getVoidTy(M->getContext());
    Type *floatType  = Type::getFloatTy(M->getContext());
    Type *floatPtrType = PointerType::getUnqual(floatType);
    Type *intType = Type::getInt64Ty(M->getContext());

    llvm::FunctionCallee Callee = M->getOrInsertFunction(
      getImplName(), voidType, intType, floatPtrType, floatPtrType, floatPtrType);
    llvm::Function *F = llvm::dyn_cast<llvm::Function>(Callee.getCallee());
    assert (F && "We must get a function");

    Value *vc = genLoc(LC);
    Value *va = genLoc(LA);
    Value *vb = genLoc(LB);

    CallInst *callInst = Builder.CreateCall(F, {getN(Builder), vc, va, vb});
    callInst->setDebugLoc(Loc);
  }

  virtual std::string getImplName() {
    return "impl_dotprod_n";
  }

  virtual Value* getN(PollyIRBuilder &Builder) {
    return Builder.getInt64(SL);
  }
};

class DotProdIntrinsicEmitter : public ReplacementEmitter {
  public:

  Instruction *LC;
  Instruction *LA;
  Instruction *LB;

  DotProdIntrinsicEmitter(ScopStmt &Stmt,
                          Instruction *LC,
                          Instruction *LA,
                          Instruction *LB)
    : ReplacementEmitter(Stmt), LC(LC), LA(LA), LB(LB) {}

  void emit(PollyIRBuilder &Builder,
            const DebugLoc &Loc,
            GenerateLocation genLoc) override {
    Module *M = Stmt.getParent()->getFunction().getParent();

    llvm::Function *F = llvm::Intrinsic::getDeclaration(M, Intrinsic::x86_sse41_dpps); // 4 wide
    // llvm::Function *F = llvm::Intrinsic::getDeclaration(M, Intrinsic::x86_avx_dp_ps_256); // 8 wide

    Type *floatScalarType = Type::getFloatTy(M->getContext());
    VectorType *floatVectorType = VectorType::get(floatScalarType, INTRIWIDTH, false);

    Value *va = genLoc(LA);
    LoadInst *LoadAVector =
      Builder.CreateLoad(floatVectorType, va);
    Value *vb = genLoc(LB);
    LoadInst *LoadBVector =
      Builder.CreateLoad(floatVectorType, vb);

    Value* allOnes = Builder.getInt8(-1);
    CallInst *dotProdResult = Builder.CreateCall(F, {LoadAVector, LoadBVector, allOnes});
    dotProdResult->setDebugLoc(Loc);

    Value *vc = genLoc(LC);
    LoadInst *LoadC =
      Builder.CreateLoad(floatScalarType, vc);

    Value *dotProdResultScalar =
      Builder.CreateExtractElement(dotProdResult, Builder.getInt64(0));
    Value *addResult = Builder.CreateFAdd(LoadC, dotProdResultScalar);

    StoreInst *StoreC =
      Builder.CreateStore(addResult, vc);
  }
};

class DotProdNEmitter : public DotProdEmitter {
public:

  Value *n;

  DotProdNEmitter(ScopStmt &Stmt,
                  Instruction *LC,
                  Instruction *LA,
                  Instruction *LB)
    : DotProdEmitter(Stmt, LC, LA, LB) {}

  virtual void addUpperBound(Value *ub) override {
    n = ub;
  }

  virtual Value* getN(PollyIRBuilder &Builder) {
    return n;
  }
};

class VectMatMulEmitter : public ReplacementEmitter {
public:
  Instruction *LC;
  Instruction *LA;
  Instruction *LB;

  Value *Beta = nullptr;
  Value *strideB = nullptr;

  std::vector<Value *> bounds;

  VectMatMulEmitter(ScopStmt &Stmt,
                      Instruction *LC,
                      Instruction *LA,
                      Instruction *LB,
                      Value *Beta,
                      Value *strideB)
    : ReplacementEmitter(Stmt), LC(LC), LA(LA), LB(LB), Beta(Beta), strideB(strideB) {}

  void emit(PollyIRBuilder &Builder,
            const DebugLoc &Loc,
            GenerateLocation genLoc) override {
    Module *M = Stmt.getParent()->getFunction().getParent();
    Type *voidType  = Type::getVoidTy(M->getContext());
    Type *floatType = Type::getFloatTy(M->getContext());
    Type *floatPtrType = PointerType::getUnqual(floatType);
    Type *longType = Type::getInt64Ty(M->getContext());

    llvm::FunctionCallee Callee = M->getOrInsertFunction(
      getImplName(), voidType, longType, longType, floatPtrType, floatPtrType, floatPtrType, longType, floatType);
    llvm::Function *F = llvm::dyn_cast<llvm::Function>(Callee.getCallee());
    assert (F && "We must get a function");

    Value *vc = genLoc(LC);
    Value *va = genLoc(LA);
    Value *vb = genLoc(LB);

    Value *bm = getM(Builder);
    Value *bn = getN(Builder);

    if ( ! Beta)
      Beta = ConstantFP::get(floatType, 1.0);

    if ( ! strideB)
      strideB = bm;

    CallInst *callInst = Builder.CreateCall(F, {bm, bn, vc, va, vb, strideB, Beta});
    callInst->setDebugLoc(Loc);
  }

  std::string getImplName() {
    return "impl_vect_matmul_mn";
  }

  virtual Value *getM(PollyIRBuilder &Builder) {
    return Builder.getInt64(SM);
  }

  virtual Value *getN(PollyIRBuilder &Builder) {
    return Builder.getInt64(SN);
  }
};

class VectMatMulMNEmitter : public VectMatMulEmitter {
public:
  Instruction *LC;
  Instruction *LA;
  Instruction *LB;

  Value *strideB;

  VectMatMulMNEmitter(ScopStmt &Stmt,
                      Instruction *LC,
                      Instruction *LA,
                      Instruction *LB,
                      Value *Beta,
                      Value *strideB)
    : VectMatMulEmitter(Stmt, LC, LA, LB, Beta, strideB) {}

  virtual void addUpperBound(Value *ub) override {
    bounds.push_back(ub);
  }

  virtual Value *getM(PollyIRBuilder &Builder) override {
    return bounds[0];
  }

  virtual Value *getN(PollyIRBuilder &Builder) override {
    return bounds[1];
  }
};

class MatMulEmitter : public ReplacementEmitter {
public:
  Instruction *LC;
  Instruction *LA;
  Instruction *LB;

  Value *Alpha = nullptr;
  Value *strideA = nullptr;
  Value *strideB = nullptr;

  std::vector<Value *> bounds;

  MatMulEmitter(ScopStmt &Stmt,
                Instruction *LC,
                Instruction *LA,
                Instruction *LB,
                Value *strideA,
                Value *strideB,
                Value *Alpha)
    : ReplacementEmitter(Stmt), LC(LC), LA(LA), LB(LB), strideA(strideA), strideB(strideB), Alpha(Alpha) {}

  void emit(PollyIRBuilder &Builder,
            const DebugLoc &Loc,
            GenerateLocation genLoc) override {
    Module *M = Stmt.getParent()->getFunction().getParent();
    Type *voidType  = Type::getVoidTy(M->getContext());
    Type *floatType  = Type::getFloatTy(M->getContext());
    Type *floatPtrType = PointerType::getUnqual(floatType);
    Type *longType = Type::getInt64Ty(M->getContext());

    llvm::FunctionCallee Callee = M->getOrInsertFunction(
      getImplName(), voidType, longType, longType, longType, floatPtrType, floatPtrType, floatPtrType, longType, longType, floatType);
    llvm::Function *F = llvm::dyn_cast<llvm::Function>(Callee.getCallee());
    assert (F && "We must get a function");

    Value *bl = getL(Builder);
    Value *bm = getM(Builder);
    Value *bn = getN(Builder);

    Value *vc = genLoc(LC);
    Value *va = genLoc(LA);
    Value *vb = genLoc(LB);

    if ( ! Alpha)
      Alpha = ConstantFP::get(floatType, 1.0);
    if ( ! strideA)
      strideA = bn;
    if ( ! strideB)
      strideB = bm;

    CallInst *callInst = Builder.CreateCall(F, {bl, bm, bn, vc, va, vb, strideA, strideB, Alpha});
    callInst->setDebugLoc(Loc);
  }

  std::string getImplName() {
    return "impl_matmul_lmn";
  }

  virtual Value *getL(PollyIRBuilder &Builder) {
    return Builder.getInt64(SL);
  }

  virtual Value *getM(PollyIRBuilder &Builder) {
    return Builder.getInt64(SM);
  }

  virtual Value *getN(PollyIRBuilder &Builder) {
    return Builder.getInt64(SN);
  }
};

class MatMulSMEEmitter : public ReplacementEmitter {
public:
  Instruction *LC;
  Instruction *LA;
  Instruction *LB;

  Value *strideA = nullptr;
  Value *strideB = nullptr;

  MatMulSMEEmitter(ScopStmt &Stmt,
                   Instruction *LC,
                   Instruction *LA,
                   Instruction *LB,
                   Value *strideA,
                   Value *strideB)
    : ReplacementEmitter(Stmt), LC(LC), LA(LA), LB(LB), strideA(strideA), strideB(strideB) {}

  void emit(PollyIRBuilder &Builder,
            const DebugLoc &Loc,
            GenerateLocation genLoc) override {
    Module *M = Stmt.getParent()->getFunction().getParent();
    Type *voidType  = Type::getVoidTy(M->getContext());
    Type *intType = Type::getInt32Ty(M->getContext());
    Type *intPtrType = PointerType::getUnqual(intType);

    llvm::FunctionCallee Callee = M->getOrInsertFunction(
      getImplName(), voidType, intType, intType, intType, intPtrType, intPtrType, intPtrType, intType, intType);
    llvm::Function *F = llvm::dyn_cast<llvm::Function>(Callee.getCallee());
    assert (F && "We must get a function");

    Value *bl = getL(Builder);
    Value *bm = getM(Builder);
    Value *bn = getN(Builder);

    Value *vc = genLoc(LC);
    Value *va = genLoc(LA);
    Value *vb = genLoc(LB);

    if ( ! strideA)
      strideA = bn;
    if ( ! strideB)
      strideB = bm;

    CallInst *callInst = Builder.CreateCall(F, {bl, bm, bn, vc, va, vb, strideA, strideB});
    callInst->setDebugLoc(Loc);
  }

  std::string getImplName() {
    return "impl_matmul_sme";
  }

  virtual Value *getL(PollyIRBuilder &Builder) {
    return Builder.getInt32(SME_L);
  }

  virtual Value *getM(PollyIRBuilder &Builder) {
    return Builder.getInt32(SME_M);
  }

  virtual Value *getN(PollyIRBuilder &Builder) {
    return Builder.getInt32(SME_N);
  }
};

class MatMulAMXEmitter : public ReplacementEmitter {
public:
  Instruction *LC;
  Instruction *LA;
  Instruction *LB;

  Value *strideA = nullptr;
  Value *strideB = nullptr;

  MatMulAMXEmitter(ScopStmt &Stmt,
                   Instruction *LC,
                   Instruction *LA,
                   Instruction *LB,
                   Value *strideA,
                   Value *strideB)
    : ReplacementEmitter(Stmt), LC(LC), LA(LA), LB(LB), strideA(strideA), strideB(strideB) {}

  void emit(PollyIRBuilder &Builder,
            const DebugLoc &Loc,
            GenerateLocation genLoc) override {
    Module *M = Stmt.getParent()->getFunction().getParent();
    Type *voidType  = Type::getVoidTy(M->getContext());
    Type *intType = Type::getInt32Ty(M->getContext());
    Type *intPtrType = PointerType::getUnqual(intType);

    llvm::FunctionCallee Callee = M->getOrInsertFunction(
      getImplName(), voidType, intType, intType, intType, intPtrType, intPtrType, intPtrType, intType, intType);
    llvm::Function *F = llvm::dyn_cast<llvm::Function>(Callee.getCallee());
    assert (F && "We must get a function");

    Value *bl = getL(Builder);
    Value *bm = getM(Builder);
    Value *bn = getN(Builder);

    Value *vc = genLoc(LC);
    Value *va = genLoc(LA);
    Value *vb = genLoc(LB);

    if ( ! strideA)
      strideA = bn;
    if ( ! strideB)
      strideB = bm;

    CallInst *callInst = Builder.CreateCall(F, {bl, bm, bn, vc, va, vb, strideA, strideB});
    callInst->setDebugLoc(Loc);
  }

  std::string getImplName() {
    return "impl_matmul_amx";
  }

  virtual Value *getL(PollyIRBuilder &Builder) {
    return Builder.getInt32(AMX_L);
  }

  virtual Value *getM(PollyIRBuilder &Builder) {
    return Builder.getInt32(AMX_M);
  }

  virtual Value *getN(PollyIRBuilder &Builder) {
    return Builder.getInt32(AMX_N);
  }
};

class MatMulLMNEmitter : public MatMulEmitter {
public:
  MatMulLMNEmitter(ScopStmt &Stmt,
                   Instruction *LC,
                   Instruction *LA,
                   Instruction *LB,
                   Value *strideA,
                   Value *strideB,
                   Value *Alpha)
    : MatMulEmitter(Stmt, LC, LA, LB, strideA, strideB, Alpha) {}

  virtual void addUpperBound(Value *ub) override {
    bounds.push_back(ub);
  }

  virtual Value *getL(PollyIRBuilder &Builder) override {
    return bounds[0];
  }

  virtual Value *getM(PollyIRBuilder &Builder) override {
    return bounds[1];
  }

  virtual Value *getN(PollyIRBuilder &Builder) override {
    return bounds[2];
  }
};

class MatMulInterchangedLMNEmitter : public MatMulLMNEmitter {
public:
  MatMulInterchangedLMNEmitter(ScopStmt &Stmt,
                               Instruction *LC,
                               Instruction *LA,
                               Instruction *LB,
                               Value *strideA,
                               Value *strideB,
                               Value *Alpha)
    : MatMulLMNEmitter(Stmt, LC, LA, LB, strideA, strideB, Alpha) {}

  virtual Value *getL(PollyIRBuilder &Builder) override {
    return bounds[0];
  }

  virtual Value *getM(PollyIRBuilder &Builder) override {
    return bounds[2]; // Interchange
  }

  virtual Value *getN(PollyIRBuilder &Builder) override {
    return bounds[1]; // Interchange
  }
};

class MatMulInterchangedEmitter : public MatMulLMNEmitter {
public:
  MatMulInterchangedEmitter(ScopStmt &Stmt,
                            Instruction *LC,
                            Instruction *LA,
                            Instruction *LB,
                            Value *strideA,
                            Value *strideB,
                            Value *Alpha)
    : MatMulLMNEmitter(Stmt, LC, LA, LB, strideA, strideB, Alpha) {}

  virtual Value *getL(PollyIRBuilder &Builder) override {
    return Builder.getInt64(SL);
  }

  virtual Value *getM(PollyIRBuilder &Builder) override {
    return Builder.getInt64(SN); // Interchange
  }

  virtual Value *getN(PollyIRBuilder &Builder) override {
    return Builder.getInt64(SM); // Interchange
  }
};

class AxpyEmitter : public ReplacementEmitter {
  public:

  Instruction *LC;
  Instruction *LA;
  Instruction *LB;

  AxpyEmitter(ScopStmt &Stmt,
              Instruction *LC,
              Instruction *LA,
              Instruction *LB)
    : ReplacementEmitter(Stmt), LC(LC), LA(LA), LB(LB) {}

  void emit(PollyIRBuilder &Builder,
            const DebugLoc &Loc,
            GenerateLocation genLoc) override {
    Module *M = Stmt.getParent()->getFunction().getParent();
    Type *voidType  = Type::getVoidTy(M->getContext());
    Type *floatType  = Type::getFloatTy(M->getContext());
    Type *floatPtrType = PointerType::getUnqual(floatType);
    Type *longType = Type::getInt64Ty(M->getContext());

    llvm::FunctionCallee Callee = M->getOrInsertFunction(
      getImplName(),
      voidType,
      longType, floatType, floatPtrType, longType, floatPtrType, longType);
    llvm::Function *F = llvm::dyn_cast<llvm::Function>(Callee.getCallee());
    assert (F && "We must get a function");

    Value *vc = genLoc(LC);
    Value *va = genLoc(LA);
    Value *vb = genLoc(LB);
    LoadInst *LoadB =
      Builder.CreateLoad(floatType, vb);
    Value *one = Builder.getInt64(1);

    CallInst *callInst = Builder.CreateCall(F, {getN(Builder), LoadB, va, one, vc, one});
    callInst->setDebugLoc(Loc);
  }

  std::string getImplName() {
    return "cblas_saxpy";
  }

  virtual Value* getN(PollyIRBuilder &Builder) {
    return Builder.getInt64(SL);
  }
};

class AxpyNEmitter : public AxpyEmitter {
  public:
  AxpyNEmitter(ScopStmt &Stmt,
               Instruction *LC,
               Instruction *LA,
               Instruction *LB)
    : AxpyEmitter(Stmt, LC, LA, LB) {}

  Value *n;

  virtual void addUpperBound(Value *ub) override {
    n = ub;
  }

  virtual Value* getN(PollyIRBuilder &Builder) override {
    return n;
  }
};

} // namespace polly

namespace {

class ScopMatcher : public ScopPass {
public:
  static char ID;
  std::vector<std::string> NewAccessStrings;
  explicit ScopMatcher() : ScopPass(ID) {}

  bool runOnScop(Scop &S) override;

  void printScop(raw_ostream &OS, Scop &S) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

class StmtPattern {
public:
  StmtPattern(const Dependences &D) : D(D) {}

  const Dependences &D;

  bool match(ScopStmt &Stmt) {
    auto matchedInstructions = matchInstructions(Stmt.getInstructions());
    if ( ! matchedInstructions)
      return false;

    bool accessesSucessfullyMatched = matchAccesses(Stmt, matchedInstructions.value());
    if ( ! accessesSucessfullyMatched) {
      LLVM_DEBUG(dbgs() << "accessesSucessfullyMatched is false\n");
      return false;
    }

    return adaptSchedule(Stmt);
  }
protected:
  // FIXME This is used to set the stride in cases where it has to be computed in matchSingleAccess.
  //       Make this more general.
  //       Zero means: not set
  long preliminaryStride = 0;

private:
  virtual std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) = 0;
  virtual unsigned int getNodeCount() = 0;
  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) = 0;
  virtual bool enableTiling() { return false; }
  // FIXME This could be generalized,
  //       but for the time being,
  //       we use only two modes:
  //       All dimensions are fixed, or
  //       all dimensions are parameterized.
  // Specify parameterized dimensions by setting them to 0.
  virtual std::vector<unsigned> getDimensions() = 0;

  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) {}

  std::optional<std::map<Instruction *, const char *>> matchInstructions(const std::vector<Instruction *> Instructions) {
    if (getNodeCount() != Instructions.size()) {
      LLVM_DEBUG(dbgs() << "Node count does not match\n");
      return std::nullopt;
    }

    // Maybe it is enough to just look at the last instruction
    for (Instruction *I : Instructions) {
      auto mi = matchSingleInstruction(I);
      if (mi)
        return mi;
    }
    LLVM_DEBUG(dbgs() << "Instruction pattern does not match\n");
    return std::nullopt;
  }

  // FIXME Also check that arrays with the same name have the same pointer.
  //       I.e. Same array ids in pattern also *must* have the same in toMatch.
  //            The reverse is *not* true. Same array ids in toMatch *can* have different
  //            array ids in pattern. Implement it with a map lookup.
  //            patternId -> toMatchId
  //            If patternId is not yet in the map, insert it.
  //            If patternId is in the map, check if the toMatchIds are identical.
  // Is it necessary to distinguish between writes and reads?
  bool matchAccesses(const ScopStmt &Stmt, std::map<Instruction *, const char *> matchedInstructions) {
    LLVM_DEBUG(dbgs() << "Begin matchAccesses\n");
    bool foundMatch = true;
    for (auto Access : Stmt) {

      if ( ! Access->isArrayKind())
        continue;

      Instruction *I = Access->getAccessInstruction();
      auto lookupResult = matchedInstructions.find(I);
      if (lookupResult == matchedInstructions.end())
        llvm_unreachable("All accesses have to be found");

      if ( ! matchSingleAccess(Access->getAccessRelation(), lookupResult->second, Stmt.getIslCtx())) {
        LLVM_DEBUG(dbgs() << "End matchAccesses: not matched\n");
        return false;
      }

      const ScopArrayInfo *SAI = Access->getScopArrayInfo();
      ScalarEvolution *SE = Stmt.getParent()->getSE();
      setStride(Stmt, I, SAI, SE);
    }

    LLVM_DEBUG(dbgs() << "End matchAccesses: all matched\n");
    return true;
  }

  bool matchSingleAccess(const isl::map &toMatchMap, const char *pattern_string, isl::ctx ctx) {
    int equal = 0;

    isl::map pattern(ctx, pattern_string);
    LLVM_DEBUG(dbgs() << "pattern is " << stringFromIslObj(pattern) << "\n");

    isl::space patternSpace = pattern.space();
    LLVM_DEBUG(dbgs() << "patternSpace is " << stringFromIslObj(patternSpace) << "\n");

    unsigned patternDimIn = unsignedFromIslSize(patternSpace.dim(isl::dim::in));
    LLVM_DEBUG(dbgs() << "patternDimIn is " << patternDimIn << "\n");
    unsigned patternDimOut = unsignedFromIslSize(patternSpace.dim(isl::dim::out));
    LLVM_DEBUG(dbgs() << "patternDimOut is " << patternDimOut << "\n");

    isl::map toMatch(ctx, stringFromIslObj(toMatchMap).c_str());
    LLVM_DEBUG(dbgs() << "toMatch is " << stringFromIslObj(toMatch) << "\n");

    isl::space toMatchSpace = toMatch.space();
    LLVM_DEBUG(dbgs() << "toMatchSpace is " << stringFromIslObj(toMatchSpace) << "\n");

    unsigned toMatchDimIn = unsignedFromIslSize(toMatchSpace.dim(isl::dim::in));
    LLVM_DEBUG(dbgs() << "toMatchDimIn is " << toMatchDimIn << "\n");
    unsigned toMatchDimOut = unsignedFromIslSize(toMatchSpace.dim(isl::dim::out));
    LLVM_DEBUG(dbgs() << "toMatchDimOut is " << toMatchDimOut << "\n");


    // Try to get out the correct dimensions if the out dim is flattened.
    // And remove constant summands. Special cases for one and two dimensions.
    // FIXME Make sure toMatchPwma and toMatchMa each have only one piece.
    if (toMatchDimOut == 1) {
      isl::pw_multi_aff toMatchPwma = isl::pw_multi_aff::from_map(toMatch);
      isl::multi_aff toMatchMa;
      const std::function get_multi_aff = [&toMatchMa](isl::set, isl::multi_aff ma) {
          toMatchMa = ma;
          return isl::stat::ok();
        };
      toMatchPwma.foreach_piece(get_multi_aff);
      isl::aff toMatchAff = toMatchMa.at(0);
      LLVM_DEBUG(dbgs() << "toMatchAff is " << stringFromIslObj(toMatchAff) << "\n");

      std::vector<long> numerators(toMatchDimIn);
      std::vector<long> denominators(toMatchDimIn);
      for (int d = 0; d < toMatchDimIn; ++d) {
        isl::val coefficient = toMatchAff.get_coefficient_val(isl::dim::in, d);
        numerators[d] = coefficient.num_si();
        denominators[d] = coefficient.den_si();
        LLVM_DEBUG(dbgs() << "numerator is " << coefficient.num_si() << "\n");
      }
      std::function isNonZero = [](long x){return x != 0;};
      int numConsideredDims = std::min(patternDimIn, toMatchDimIn);
      int numNonZero = std::count_if(numerators.end() - numConsideredDims, numerators.end(), isNonZero);
      LLVM_DEBUG(dbgs() << "numNonZero is " << numNonZero << "\n");

      isl::set toMatchDomain = toMatch.domain();
      isl::space toMatchDomainSpace = toMatchDomain.space();
      isl::multi_aff prepare_dims = toMatchMa.identity_on_domain(toMatchDomainSpace);

      // One dimensional array
      if (numNonZero == 1) {
        auto dimIter = std::find_if(numerators.end() - numConsideredDims, numerators.end(), isNonZero);
        int onlyNonZeroIndex = std::distance(numerators.begin(), dimIter);
        LLVM_DEBUG(dbgs() << "onlyNonZeroIndex is " << onlyNonZeroIndex << "\n");

        // Conservatively only allow step size one
        if (numerators[onlyNonZeroIndex] == 1 && denominators[onlyNonZeroIndex] == 1) {
          isl::pw_multi_aff single_dim_pw = prepare_dims.drop_dims(isl::dim::out, 0, onlyNonZeroIndex);
          if (onlyNonZeroIndex < toMatchDimIn - 1) {
            single_dim_pw = single_dim_pw.drop_dims(isl::dim::out, 1, toMatchDimIn - 1 - onlyNonZeroIndex);
          }
          // Change the toMatch map for comparison
          toMatch = single_dim_pw.as_map();
          LLVM_DEBUG(dbgs() << "new toMatch is " << stringFromIslObj(toMatch) << "\n");
        }
      }

      // 2D matrix
      if (numNonZero == 2) {
        auto dimOuter = std::find_if(numerators.end() - numConsideredDims, numerators.end(), isNonZero);
        int outerNonZeroIndex = std::distance(numerators.begin(), dimOuter);
        LLVM_DEBUG(dbgs() << "outerNonZeroIndex is " << outerNonZeroIndex << "\n");
        auto dimInner = std::find_if(dimOuter + 1, numerators.end(), isNonZero);
        int innerNonZeroIndex = std::distance(numerators.begin(), dimInner);
        LLVM_DEBUG(dbgs() << "innerNonZeroIndex is " << innerNonZeroIndex << "\n");

        long outerNumerator = numerators[outerNonZeroIndex];
        long outerDenominator = denominators[outerNonZeroIndex];
        long innerNumerator = numerators[innerNonZeroIndex];
        long innerDenominator = denominators[innerNonZeroIndex];

        isl::space toMatchSpace = toMatch.space();
        // Expand to two output dimensions
        toMatchSpace = toMatchSpace.add_dims(isl::dim::out, 1);
        isl::map matr = toMatchSpace.universe_map();
        LLVM_DEBUG(dbgs() << "matr is " << stringFromIslObj(matr) << "\n");

        // Non transposed case
        if (outerDenominator == 1 && innerDenominator == 1 && outerNumerator >= 1 && innerNumerator == 1) {
          LLVM_DEBUG(dbgs() << "Non transposed case\n");
          matr = matr.equate(isl::dim::in, toMatchDimIn - 2, isl::dim::out, 0);
          matr = matr.equate(isl::dim::in, toMatchDimIn - 1, isl::dim::out, 1);
          LLVM_DEBUG(dbgs() << "matr is " << stringFromIslObj(matr) << "\n");

          // Change the toMatch map for comparison
          toMatch = matr;
          toMatchDimOut = 2;
          LLVM_DEBUG(dbgs() << "new toMatch is " << stringFromIslObj(toMatch) << "\n");
          preliminaryStride = outerNumerator;
          LLVM_DEBUG(dbgs() << "preliminaryStride is " << preliminaryStride << "\n");
        }
        // Transposed case
        else if (outerDenominator == 1 && innerDenominator == 1 && outerNumerator == 1 && innerNumerator >= 1) {
          LLVM_DEBUG(dbgs() << "Transposed case\n");
          // Here is the swap
          matr = matr.equate(isl::dim::in, toMatchDimIn - 2, isl::dim::out, 1);
          matr = matr.equate(isl::dim::in, toMatchDimIn - 1, isl::dim::out, 0);
          LLVM_DEBUG(dbgs() << "matr is " << stringFromIslObj(matr) << "\n");

          // Change the toMatch map for comparison
          toMatch = matr;
          toMatchDimOut = 2;
          LLVM_DEBUG(dbgs() << "new toMatch is " << stringFromIslObj(toMatch) << "\n");
          preliminaryStride = innerNumerator;
          LLVM_DEBUG(dbgs() << "preliminaryStride is " << preliminaryStride << "\n");
        }
      }
    }

    // Dimensions must be compatible
    if ( ! (  patternDimOut <= toMatchDimOut
           && patternDimIn  <= toMatchDimIn)) {
      LLVM_DEBUG(dbgs() << "matchSingleAccess end: Dimensions do not match\n");
      return false;
    }

    // Most operations remove the ids from pattern.
    // I do not know why.
    // Maybe this is not a problem, and we can work
    // without the id.
    isl::id patternInId  = pattern.get_tuple_id(isl::dim::in);
    isl::id patternOutId = pattern.get_tuple_id(isl::dim::out);
    LLVM_DEBUG(dbgs() << "patternOutId is " << stringFromIslObj(patternOutId) << "\n");

    pattern = pattern.insert_dims(isl::dim::in,  0, toMatchDimIn  - patternDimIn);
    pattern = pattern.insert_dims(isl::dim::out, 0, toMatchDimOut - patternDimOut);

    pattern = pattern.drop_constraints_not_involving_dims(isl::dim::in,
                                                          toMatchDimIn - patternDimIn,
                                                          patternDimIn);

    pattern = pattern.set_tuple_id(isl::dim::in,  patternInId);
    pattern = pattern.set_tuple_id(isl::dim::out, patternOutId);
    LLVM_DEBUG(dbgs() << "pattern is " << stringFromIslObj(pattern) << "\n");


    toMatch = toMatch.drop_constraints_not_involving_dims(isl::dim::in,
                                                          toMatchDimIn - patternDimIn,
                                                          patternDimIn);
    toMatch = toMatch.set_tuple_id(isl::dim::in,  patternInId);
    toMatch = toMatch.set_tuple_id(isl::dim::out, patternOutId);
    LLVM_DEBUG(dbgs() << "toMatch is " << stringFromIslObj(toMatch) << "\n");

    equal = pattern.is_equal(toMatch);
    LLVM_DEBUG(dbgs() << "matchSingleAccess end: equal is " << equal << "\n");

    return equal;
  }

  void print_node_type(isl::schedule_node n) {
    if (n.isa<isl::schedule_node_band>())
      LLVM_DEBUG(dbgs() << "schedule_node_band\n");
    if (n.isa<isl::schedule_node_context>())
      LLVM_DEBUG(dbgs() << "schedule_node_context\n");
    if (n.isa<isl::schedule_node_domain>())
      LLVM_DEBUG(dbgs() << "schedule_node_domain\n");
    if (n.isa<isl::schedule_node_expansion>())
      LLVM_DEBUG(dbgs() << "schedule_node_expansion\n");
    if (n.isa<isl::schedule_node_extension>())
      LLVM_DEBUG(dbgs() << "schedule_node_extension\n");
    if (n.isa<isl::schedule_node_filter>())
      LLVM_DEBUG(dbgs() << "schedule_node_filter\n");
    if (n.isa<isl::schedule_node_guard>())
      LLVM_DEBUG(dbgs() << "schedule_node_guard\n");
    if (n.isa<isl::schedule_node_leaf>())
      LLVM_DEBUG(dbgs() << "schedule_node_leaf\n");
    if (n.isa<isl::schedule_node_mark>())
      LLVM_DEBUG(dbgs() << "schedule_node_mark\n");
    if (n.isa<isl::schedule_node_sequence>())
      LLVM_DEBUG(dbgs() << "schedule_node_sequence\n");
    if (n.isa<isl::schedule_node_set>())
      LLVM_DEBUG(dbgs() << "schedule_node_set\n");
  }

  // place is the output parameter
  static void check_node(isl::schedule_node n, isl::schedule_node &place, std::string name) {
    // print_node_type(n);

    // We only look for leaf nodes
    if ( ! n.isa<isl::schedule_node_leaf>())
      return;

    assert(n.has_parent() && "Leaf node must have a parent");

    isl::union_set domain = n.domain();
    // This has to be a set since we are looking at only one leaf
    isl::set domainSet = domain.as_set();
    std::string leafName = domainSet.tuple_name();
    if (name == leafName)
      place = n;
  }

  // // This is equivalent to foreach_descendant_top_down.
  // void traverse(isl::schedule_node n) {
  //   if ( ! n.has_children() && ! n.has_next_sibling())
  //     return;
  //   if (n.has_children())
  //     traverse(n.first_child());
  //   if (n.has_next_sibling())
  //     traverse(n.next_sibling());
  // }

  bool adaptSchedule(ScopStmt &Stmt) {
    isl::ctx ctx = Stmt.getIslCtx();
    isl::schedule ScheduleTree = Stmt.getParent()->getScheduleTree();
    LLVM_DEBUG(dbgs() << "ScheduleTree is " << stringFromIslObj(ScheduleTree) << "\n");
    isl::schedule_node scheduleTreeRoot = ScheduleTree.root();
    isl::union_set domain = scheduleTreeRoot.as<isl::schedule_node_domain>().domain();

    // Lookup the leaf node.
    isl::schedule_node leaf = scheduleTreeRoot;
    std::string leafName = Stmt.getDomainId().name();
    scheduleTreeRoot.foreach_descendant_top_down([&leaf, leafName](isl::schedule_node n) {
        check_node(n, leaf, leafName);
        return true;
      });

    std::vector<unsigned> dimensions = getDimensions();
    unsigned replacedDimensionsCount = dimensions.size();

    // Find the outermost relevant band node.
    // Collect sequence nodes.
    unsigned countDims = 0;
    isl::schedule_node place = leaf;

    std::vector<isl::union_set_list> sequenceFilters;
    while ( ! place.isa<isl::schedule_node_domain>()) {
      assert(place.has_parent() && "Node must have a parent if it is not the domain node.");
      if ( place.isa<isl::schedule_node_band>()) {
        countDims++;
        if (countDims == replacedDimensionsCount)
          break;
      }

      if (place.isa<isl::schedule_node_sequence>()) {
        LLVM_DEBUG(dbgs() << "Walking up, found sequence node\n");
        if (place.first_child().isa<isl::schedule_node_filter>()) {
          LLVM_DEBUG(dbgs() << "  place has " << unsignedFromIslSize(place.n_children())  << " children\n");

          isl::schedule_node_filter snf = place.first_child().as<isl::schedule_node_filter>();
          isl::union_set snf_fi(snf.get_filter());
          isl::union_set_list fl = snf_fi.to_list();
          while (snf.has_next_sibling()) {
            snf = snf.next_sibling().as<isl::schedule_node_filter>();
            snf_fi = snf.get_filter();
            fl = fl.add(snf_fi);
          }
          sequenceFilters.push_back(fl);
        }
      }

      place = place.parent();
    }

    assert (place.isa<isl::schedule_node_band>() && "Replacement place must be a band node");


    isl::schedule_node pp = place.parent();
    for (int i = 0; i < sequenceFilters.size(); ++i) {
      pp = pp.insert_sequence(sequenceFilters[i]);
    }


    // Find the correct place again.
    scheduleTreeRoot = pp.root();
    scheduleTreeRoot.foreach_descendant_top_down([&leaf, leafName](isl::schedule_node n) {
        check_node(n, leaf, leafName);
        return true;
      });

    place = leaf;
    countDims = 0;
    while ( ! place.isa<isl::schedule_node_domain>()) {
      if ( place.isa<isl::schedule_node_band>()) {
        countDims++;
        if (countDims == replacedDimensionsCount)
          break;
      }

      assert( ! place.isa<isl::schedule_node_sequence>() &&
          "All sequence nodes should have been move out of the way");

      place = place.parent();
    }

    assert (place.isa<isl::schedule_node_band>() && "Replacement place must be a band node");


    // FIXME put the mark string into the header
    isl::id user_mark = isl::id::alloc(ctx, "REPLACE_LOOP", getEmitter(Stmt));
    isl::schedule_node newRoot;
    if (enableTiling()) {
      // Collect all relevant filters
      isl::schedule_node runner = leaf;
      while ( ! runner.isa<isl::schedule_node_domain>()) {
        assert(runner.has_parent() && "Node must have a parent if it is not the domain node.");
        if ( runner.isa<isl::schedule_node_filter>()) {
          isl::union_set filter = runner.as<isl::schedule_node_filter>().get_filter();
          domain = domain.intersect(filter);
        }
        runner = runner.parent();
      }

      // This has to be a set since we are looking at only one leaf
      isl::set domainSet = domain.as_set();
      LLVM_DEBUG(dbgs() << "domainSet is " << stringFromIslObj(domainSet) << "\n");
      isl::space domainSpace = domainSet.space();


      unsigned setDim = unsignedFromIslSize(domainSet.dim(isl::dim::set));
      // innermostDim is the Dimension that would be innermost after replacement
      unsigned innermostDim = setDim - replacedDimensionsCount;

      isl::basic_set_list basicDomainSetList = domainSet.basic_set_list();
      assert(unsignedFromIslSize(basicDomainSetList.size()) == 1 &&
          "There must be only a single basic_set in the remaining domain");
      isl::basic_set basicDomainSet = basicDomainSetList.at(0);

      // This calls clears the ISL_BASIC_SET_FINAL flag of the basic_set
      // Then foreach_constraint fails.
      // basicDomainSet = basicDomainSet.drop_constraints_not_involving_dims(isl::dim::set, innermostDim, 1);
      LLVM_DEBUG(dbgs() << "basicDomainSet is " << stringFromIslObj(basicDomainSet) << "\n");

      // Find the bounds.
      isl_constraint_list *clist =
        isl_basic_set_get_constraint_list(basicDomainSet.get());

      // What is the difference between the two?
      unsigned nc = unsignedFromIslSize(isl::manage(isl_constraint_list_n_constraint(clist)));
      LLVM_DEBUG(dbgs() << "nc is " << nc << "\n");
      unsigned sc = unsignedFromIslSize(isl::manage(isl_constraint_list_size(clist)));
      LLVM_DEBUG(dbgs() << "sc is " << sc << "\n");

      // FIXME get minimal upper bound
      //       and maximal lower bound
      std::vector<isl::aff>bounds(replacedDimensionsCount);
      unsigned found = 0;
      for (unsigned i = 0; i < nc; ++i) {
        isl::constraint c = isl::manage(isl_constraint_list_get_constraint(clist, i));
        for (unsigned d = 0; d < replacedDimensionsCount; ++d) {
          unsigned currentDim = innermostDim + d;
          bool iub = isl_constraint_is_upper_bound(c.get(), isl_dim_set, currentDim);
          // bool ilb = isl_constraint_is_lower_bound(c.get(), isl_dim_set, innermostDim + d);
          // bool involvesDim = c.involves_dims(isl::dim::set, d, 1);
          // LLVM_DEBUG(dbgs() << "c at " << i << " iub is " << iub << "\n");
          // LLVM_DEBUG(dbgs() << "c at " << i << " ilb is " << ilb << "\n");
          // LLVM_DEBUG(dbgs() << "c at " << i << " involvesDim is " << involvesDim << "\n");
          if (iub) {
            bounds[d] = isl::manage(isl_constraint_get_bound(c.get(), isl_dim_set, currentDim));
            LLVM_DEBUG(dbgs() << "bound is " << stringFromIslObj(bounds[d]) << "\n");
            found++;
          }
        }
      }
      isl_constraint_list_free(clist);

      // FIXME This check is not enough.
      //       We have to verify every dimension separately.
      LLVM_DEBUG(dbgs() << "replacedDimensionsCount is " << replacedDimensionsCount << "\n");
      LLVM_DEBUG(dbgs() << "found is " << found << "\n");
      assert(found == replacedDimensionsCount && "We did not find enough upper bound values");

      std::vector<isl::aff>moduluses(replacedDimensionsCount);
      for (unsigned i = 0; i < replacedDimensionsCount; ++i) {
        isl::aff bound = bounds[i];
        unsigned vectorLength = dimensions[i];
        isl::aff constant = isl::aff::zero_on_domain(bound.space().domain());
        constant = constant.add_constant(vectorLength);

        isl::aff modulus = bound.add_constant(1);

        modulus = modulus.mod(constant);
        modulus = bound.sub(modulus);
        LLVM_DEBUG(dbgs() << "modulus is " << stringFromIslObj(modulus) << "\n");
        moduluses[i] = modulus;
      }

      // Split the outermost relevant loop with the moduluses.
      // The loop is split up in bulk and rest.

      isl::multi_aff ma = isl::multi_aff::identity_on_domain(domainSpace);
      LLVM_DEBUG(dbgs() << "ma is " << stringFromIslObj(ma) << "\n");
      LLVM_DEBUG(dbgs() << "ma.space is " << stringFromIslObj(ma.space()) << "\n");
      isl::map maMap = ma.as_map();
      LLVM_DEBUG(dbgs() << "maMap is " << stringFromIslObj(maMap) << "\n");

      isl::multi_aff affModulus = ma;
      for (unsigned i = 0; i < replacedDimensionsCount; ++i) {
        affModulus = affModulus.set_aff(innermostDim + i, moduluses[i]);
      }
      isl::multi_pw_aff mpaModulus(affModulus);
      LLVM_DEBUG(dbgs() << "mpaModulus is " << stringFromIslObj(mpaModulus) << "\n");


      isl::map boundedMaMap = maMap.upper_bound(mpaModulus);
      LLVM_DEBUG(dbgs() << "boundedMaMap is " << stringFromIslObj(boundedMaMap) << "\n");
      isl::set boundedDomain = boundedMaMap.domain();
      LLVM_DEBUG(dbgs() << "boundedDomain is " << stringFromIslObj(boundedDomain) << "\n");

      isl::union_set bulkFilter(boundedDomain);
      bulkFilter = bulkFilter.intersect(domain);
      LLVM_DEBUG(dbgs() << "bulkFilter is " << stringFromIslObj(bulkFilter) << "\n");

      isl::union_set restFilter = domain.subtract(bulkFilter);
      LLVM_DEBUG(dbgs() << "restFilter is " << stringFromIslObj(restFilter) << "\n");

      isl::union_set_list filterList = bulkFilter.to_list();
      filterList = filterList.add(restFilter);

      isl::schedule_node toTile = place.insert_sequence(filterList);

      // Tile the loops.
      // Tile all band nodes individually.
      // We assume there is never more than one child.
      countDims = 0;
      while ( ! toTile.isa<isl::schedule_node_leaf>()) {
        assert(toTile.has_children() && "Node must have a child.");
        if ( toTile.isa<isl::schedule_node_band>()) {
          isl::schedule_node_band toTileBand = toTile.as<isl::schedule_node_band>();
          isl::val s(ctx, dimensions[countDims]);
          countDims++;
          isl::space sizeSpace(ctx, 0, 1);
          isl::multi_val sizes(sizeSpace, s.to_list());
          toTileBand = toTileBand.tile(sizes);
          toTileBand = toTileBand.first_child().as<isl::schedule_node_band>();
          toTile = toTileBand.sink();
          toTile = toTile.parent();
          if (countDims == replacedDimensionsCount)
            break;
        }
        toTile = toTile.first_child();
      }

      isl::schedule_node markReplaceable = toTile.first_child();
      markReplaceable = markReplaceable.insert_mark(user_mark);
      newRoot = markReplaceable.root();
    } else {
      isl::schedule_node replaceWithoutTiling  = place.insert_mark(user_mark);
      newRoot = replaceWithoutTiling.root();
    }

    // FIXME This bypasses every transformation
    // isl::schedule_node newRoot = place.root();


    isl::schedule newSchedule = newRoot.get_schedule();
    LLVM_DEBUG(dbgs() << "newSchedule is " << stringFromIslObj(newSchedule) << "\n");
    LLVM_DEBUG(dbgs() << "newSchedule as map is " << stringFromIslObj(newSchedule.map()) << "\n");

    // FIXME Is there also a way to ensure that we did not
    //       drop any statement instances?
    if (D.isValidSchedule(*Stmt.getParent(), newSchedule)) {
      // If we do not violate any deps, we do the transformation.
      Stmt.getParent()->setScheduleTree(newSchedule);
      ScheduleTree = Stmt.getParent()->getScheduleTree();

      return true;
    }
    LLVM_DEBUG(dbgs() << "Schedule violates dependencies\n");

    return false;
  }
};

class DotProdStmtPattern : public StmtPattern {
public:
  DotProdStmtPattern(const Dependences &D) : StmtPattern(D) {}

  Instruction *LC = nullptr;
  Instruction *LA = nullptr;
  Instruction *LB = nullptr;
  Instruction *SC = nullptr;

  unsigned int getNodeCount() override { return 6; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new DotProdEmitter(Stmt, LC, LA, LB);
  }

  virtual std::vector<unsigned> getDimensions() override { return {SL}; }

  std::map<Instruction *, const char *> fillAccessMap() {
    std::map<Instruction *, const char *> rv;
    rv[LA] = "{ S[i] -> A[i] }";
    rv[LB] = "{ S[i] -> B[i] }";
    rv[LC] = "{ S[i] -> C[0] }";
    rv[SC] = "{ S[i] -> C[0] }";
    return rv;
  }

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    // auto P = m_Intrinsic<Intrinsic::fmuladd>(m_Value(), m_Value(), m_Value());
    Value *A;
    Value *B;
    Value *C;

    // This is for integer dotprod
    // auto P = m_Store(m_NSWAdd(m_NSWMul(m_Load(m_Value(A), LA), m_Load(m_Value(B), LB)), m_Load(m_Value(C), LC)), m_Deferred(C), SC);
    auto P0 = m_Store(m_FAdd(m_FMul(m_Load(m_Value(A), LA), m_Load(m_Value(B), LB)), m_Load(m_Value(C), LC)), m_Deferred(C), SC);
    if (PatternMatch::match(I, P0))
      return fillAccessMap();
    auto P1 = m_Store(m_FAdd(m_Load(m_Value(C), LC), m_FMul(m_Load(m_Value(A), LA), m_Load(m_Value(B), LB))), m_Deferred(C), SC);
    if (PatternMatch::match(I, P1))
      return fillAccessMap();

    return std::nullopt;
  }

  virtual bool enableTiling() override { return true; }
};

class DotProdIntrinsicStmtPattern : public DotProdStmtPattern {
public:
  DotProdIntrinsicStmtPattern(const Dependences &D) : DotProdStmtPattern(D) {}

  virtual std::vector<unsigned> getDimensions() override { return {INTRIWIDTH}; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new DotProdIntrinsicEmitter(Stmt, LC, LA, LB);
  }
};

class DotProdNStmtPattern : public DotProdStmtPattern {
public:
  DotProdNStmtPattern(const Dependences &D) : DotProdStmtPattern(D) {}

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new DotProdNEmitter(Stmt, LC, LA, LB);
  }

  // We do not need tiling if we use the parameterizable emitter.
  virtual bool enableTiling() override { return false; }
};


// This includes the generated VectMatMulStmtPattern_*
#include "ScopMatcherPatterns.inc"

class VectMatMulStmtPattern : public StmtPattern {
public:
  VectMatMulStmtPattern(const Dependences &D) : StmtPattern(D) {}

  Instruction *LC = nullptr;
  Instruction *LA = nullptr;
  Instruction *LB = nullptr;

  Value *strideA = nullptr;
  Value *strideB = nullptr;

  unsigned int getNodeCount() override { return 6; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new VectMatMulEmitter(Stmt, LC, LA, LB, nullptr, strideB);
  }

  virtual std::vector<unsigned> getDimensions() override { return {SM, SN}; }

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    Value *A, *B, *C;
    Instruction *SC;
    auto P = m_Store(
               m_c_FAdd(
                 m_c_FMul(
                   m_Load(
                     m_Value(A),
                     LA),
                   m_Load(
                     m_Value(B),
                     LB)),
                 m_Load(
                   m_Value(C),
                   LC)),
               m_Deferred(C),
               SC);

    bool r = PatternMatch::match(I, P);
    if (r) {
      std::map<Instruction *, const char *> rv;
      rv[LA] = "{ S[j,k] -> A[k] }";
      rv[LB] = "{ S[j,k] -> B[k,j] }";
      rv[LC] = "{ S[j,k] -> C[j] }";
      rv[SC] = "{ S[j,k] -> C[j] }";
      return rv;
    }
    else {
      return std::nullopt;
    }
  }

  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) override {
    if (I != LB)
      return;

    // FIXME Is there a way to ensure the dimensionIndex is adaptable and correct?
    unsigned dimensionIndex = 1;

    const SCEV *sc = SAI->getDimensionSize(dimensionIndex);
    SetVector<Value *> values;
    findValues(sc, *SE, values);

    if (values.size() > 0) {
      Value *w = values[0];
      strideB = w;
      LLVM_DEBUG(dbgs() << "strideB is " << *strideB << "\n");
      return;
    }

    if (sc->getSCEVType() == SCEVTypes::scConstant) {
      ConstantInt *intVal = static_cast<const SCEVConstant *>(sc)->getValue();

      // Ugly type conversion
      Module *M = Stmt.getParent()->getFunction().getParent();
      Type *intType = Type::getInt64Ty(M->getContext());
      strideB = ConstantInt::getSigned(intType, intVal->getSExtValue());
      LLVM_DEBUG(dbgs() << "strideB is " << *strideB << "\n");
      return;
    }

    llvm_unreachable("Cannot determine array sizes");
  }

  virtual bool enableTiling() override { return true; }
};

class VectMatMulMNStmtPattern : public VectMatMulStmtPattern {
public:
  VectMatMulMNStmtPattern(const Dependences &D) : VectMatMulStmtPattern(D) {}

  virtual bool enableTiling() override { return false; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new VectMatMulMNEmitter(Stmt, LC, LA, LB, nullptr, strideB);
  }

  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) override {
    LLVM_DEBUG(dbgs() << "setStride called\n");
    if (preliminaryStride) {
      LLVM_DEBUG(dbgs() << "preliminaryStride is " << preliminaryStride << "\n");
      Module *M = Stmt.getParent()->getFunction().getParent();
      Type *intType = Type::getInt64Ty(M->getContext());
      strideB = ConstantInt::getSigned(intType, preliminaryStride);
    }
    else {
      LLVM_DEBUG(dbgs() << "preliminaryStride not set\n");
    }
  }
};

class VectMatMulCommStmtPattern : public VectMatMulStmtPattern {
public:
  VectMatMulCommStmtPattern(const Dependences &D) : VectMatMulStmtPattern(D) {}

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    Value *A, *B, *C;
    Instruction *SC;
    auto P = m_Store(
               m_c_FAdd(
                 m_c_FMul(
                   m_Load(
                     m_Value(B),  // FIXME Here is the only difference to the superclass
                     LB),
                   m_Load(
                     m_Value(A),
                     LA)),
                 m_Load(
                   m_Value(C),
                   LC)),
               m_Deferred(C),
               SC);

    bool r = PatternMatch::match(I, P);
    if (r) {
      std::map<Instruction *, const char *> rv;
      rv[LA] = "{ S[j,k] -> A[k] }";
      rv[LB] = "{ S[j,k] -> B[k,j] }";
      rv[LC] = "{ S[j,k] -> C[j] }";
      rv[SC] = "{ S[j,k] -> C[j] }";
      return rv;
    }
    else {
      return std::nullopt;
    }
  }
};

class VectMatMulMNCommStmtPattern : public VectMatMulCommStmtPattern {
public:
  VectMatMulMNCommStmtPattern(const Dependences &D) : VectMatMulCommStmtPattern(D) {}

  virtual bool enableTiling() override { return false; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new VectMatMulMNEmitter(Stmt, LC, LA, LB, nullptr, nullptr);
  }

  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) override { }
};

class VectMatMulBetaStmtPattern : public VectMatMulStmtPattern {
public:
  VectMatMulBetaStmtPattern(const Dependences &D) : VectMatMulStmtPattern(D) {}

  Value *Beta;

  unsigned int getNodeCount() override { return 7; }

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    Value *A;
    Value *B;
    Value *C;
    Instruction *SC;
    auto P = m_Store(m_c_FAdd(m_c_FMul(m_c_FMul(m_Load(m_Value(B), LB), m_Value(Beta)), m_Load(m_Value(A), LA)), m_Load(m_Value(C), LC)), m_Deferred(C), SC);
    bool r = PatternMatch::match(I, P);
    if (r) {
      std::map<Instruction *, const char *> rv;
      rv[LA] = "{ S[j,k] -> A[k] }";
      rv[LB] = "{ S[j,k] -> B[k,j] }";
      rv[LC] = "{ S[j,k] -> C[j] }";
      rv[SC] = "{ S[j,k] -> C[j] }";
      return rv;
    }
    else {
      return std::nullopt;
    }
  }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new VectMatMulEmitter(Stmt, LC, LA, LB, Beta, strideB);
  }
};

class VectMatMulMNBetaStmtPattern : public VectMatMulBetaStmtPattern {
public:
  VectMatMulMNBetaStmtPattern(const Dependences &D) : VectMatMulBetaStmtPattern(D) {}

  virtual bool enableTiling() override { return false; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new VectMatMulMNEmitter(Stmt, LC, LA, LB, Beta, nullptr);
  }

  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) override {}
};

class MatMulStmtPattern : public StmtPattern {
public:
  MatMulStmtPattern(const Dependences &D) : StmtPattern(D) {}

  Instruction *LC = nullptr;
  Instruction *LA = nullptr;
  Instruction *LB = nullptr;

  Value *strideA = nullptr;
  Value *strideB = nullptr;

  unsigned int getNodeCount() override { return 6; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new MatMulEmitter(Stmt, LC, LA, LB, strideA, strideB, nullptr);
  }

  virtual std::vector<unsigned> getDimensions() override { return {SL, SM, SN}; }

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    // auto P = m_Intrinsic<Intrinsic::fmuladd>(m_Value(), m_Value(), m_Value());
    Value *A;
    Value *B;
    Value *C;
    // FIXME Maybe make this a member too.
    Instruction *SC;
    // This is for integer dotprod
    // auto P = m_Store(m_NSWAdd(m_NSWMul(m_Load(m_Value(A), LA), m_Load(m_Value(B), LB)), m_Load(m_Value(C), LC)), m_Deferred(C), SC);
    auto P = m_Store(m_c_FAdd(m_c_FMul(m_Load(m_Value(A), LA), m_Load(m_Value(B), LB)), m_Load(m_Value(C), LC)), m_Deferred(C), SC);
    bool r = PatternMatch::match(I, P);
    if (r) {
      std::map<Instruction *, const char *> rv;
      rv[LA] = "{ S[i,j,k] -> A[i,k] }";
      rv[LB] = "{ S[i,j,k] -> B[k,j] }";
      rv[LC] = "{ S[i,j,k] -> C[i,j] }";
      rv[SC] = "{ S[i,j,k] -> C[i,j] }";
      return rv;
    }
    else {
      return std::nullopt;
    }
  }

  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) override {
    if (I != LB && I != LA)
      return;

    unsigned dimensionIndex = 1;

    const SCEV *sc = SAI->getDimensionSize(dimensionIndex);
    SetVector<Value *> values;
    findValues(sc, *SE, values);

    if (values.size() > 0) {
      Value *w = values[0];


      if (I == LB) {
        strideB = w;
        LLVM_DEBUG(dbgs() << "strideB is " << *strideB << "\n");
      }
      else {
        strideA = w;
        LLVM_DEBUG(dbgs() << "strideA is " << *strideA << "\n");
      }

      return;
    }

    if (sc->getSCEVType() == SCEVTypes::scConstant) {
      ConstantInt *intVal = static_cast<const SCEVConstant *>(sc)->getValue();

      // Ugly type conversion
      Module *M = Stmt.getParent()->getFunction().getParent();
      Type *intType = Type::getInt64Ty(M->getContext());
      Value *s = ConstantInt::getSigned(intType, intVal->getSExtValue());
      if (I == LB) {
        strideB = s;
        LLVM_DEBUG(dbgs() << "strideB is " << *strideB << "\n");
      }
      else {
        strideA = s;
        LLVM_DEBUG(dbgs() << "strideA is " << *strideA << "\n");
      }

      return;
    }

    llvm_unreachable("Cannot determine array sizes");
  }

  virtual bool enableTiling() override { return true; }
};

class MatMulLMNStmtPattern : public MatMulStmtPattern {
public:
  MatMulLMNStmtPattern(const Dependences &D) : MatMulStmtPattern(D) {}

  virtual bool enableTiling() override { return false; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new MatMulLMNEmitter(Stmt, LC, LA, LB, nullptr, nullptr, nullptr);
  }

  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) override {}
};

class MatMulAlphaStmtPattern : public MatMulStmtPattern {
public:
  MatMulAlphaStmtPattern(const Dependences &D) : MatMulStmtPattern(D) {}

  Value *Alpha = nullptr;

  unsigned int getNodeCount() override { return 7; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new MatMulEmitter(Stmt, LC, LA, LB, strideA, strideB, Alpha);
  }

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    Value *A;
    Value *B;
    Value *C;
    Instruction *SC;
    auto P = m_Store(m_c_FAdd(m_c_FMul(m_c_FMul(m_Load(m_Value(A), LA), m_Value(Alpha)), m_Load(m_Value(B), LB)), m_Load(m_Value(C), LC)), m_Deferred(C), SC);
    bool r = PatternMatch::match(I, P);
    if (r) {
      std::map<Instruction *, const char *> rv;
      rv[LA] = "{ S[i,j,k] -> A[i,k] }";
      rv[LB] = "{ S[i,j,k] -> B[k,j] }";
      rv[LC] = "{ S[i,j,k] -> C[i,j] }";
      rv[SC] = "{ S[i,j,k] -> C[i,j] }";
      return rv;
    }
    else {
      return std::nullopt;
    }
  }
};

class MatMulAlphaLMNStmtPattern : public MatMulAlphaStmtPattern {
public:
  MatMulAlphaLMNStmtPattern(const Dependences &D) : MatMulAlphaStmtPattern (D) {}

  virtual bool enableTiling() override { return false; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new MatMulLMNEmitter(Stmt, LC, LA, LB, nullptr, nullptr, Alpha);
  }

  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) override {}
};

class MatMulAlphaInterchangedStmtPattern : public MatMulAlphaStmtPattern {
public:
  MatMulAlphaInterchangedStmtPattern(const Dependences &D) : MatMulAlphaStmtPattern(D) {}

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new MatMulInterchangedEmitter(Stmt, LC, LA, LB, strideA, strideB, Alpha);
  }

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    Value *A;
    Value *B;
    Value *C;
    Instruction *SC;
    auto P = m_Store(m_c_FAdd(m_c_FMul(m_c_FMul(m_Load(m_Value(A), LA), m_Value(Alpha)), m_Load(m_Value(B), LB)), m_Load(m_Value(C), LC)), m_Deferred(C), SC);
    bool r = PatternMatch::match(I, P);
    if (r) {
      std::map<Instruction *, const char *> rv;
      rv[LA] = "{ S[i,k,j] -> A[i,k] }";
      rv[LB] = "{ S[i,k,j] -> B[k,j] }";
      rv[LC] = "{ S[i,k,j] -> C[i,j] }";
      rv[SC] = "{ S[i,k,j] -> C[i,j] }";
      return rv;
    }
    else {
      return std::nullopt;
    }
  }
};

class MatMulAlphaInterchangedLMNStmtPattern : public MatMulAlphaInterchangedStmtPattern {
public:
  MatMulAlphaInterchangedLMNStmtPattern(const Dependences &D) : MatMulAlphaInterchangedStmtPattern (D) {}

  virtual bool enableTiling() override { return false; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new MatMulInterchangedLMNEmitter(Stmt, LC, LA, LB, nullptr, nullptr, Alpha);
  }

  virtual void setStride(const ScopStmt &Stmt, Instruction *I, const ScopArrayInfo *SAI, ScalarEvolution *SE) override {}
};

class AxpyStmtPattern : public StmtPattern {
public:
  AxpyStmtPattern(const Dependences &D)
    : StmtPattern(D) {}

  Instruction *LC = nullptr;
  Instruction *LA = nullptr;
  Instruction *LB = nullptr;

  unsigned int getNodeCount() override { return 6; }

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new AxpyEmitter(Stmt, LC, LA, LB);
  }

  virtual std::vector<unsigned> getDimensions() override { return {SL}; }

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    Value *A;
    Value *B;
    Value *C;
    Instruction *SC;
    auto P = m_Store(m_c_FAdd(m_c_FMul(m_Load(m_Value(A), LA), m_Load(m_Value(B), LB)), m_Load(m_Value(C), LC)), m_Deferred(C), SC);
    bool r = PatternMatch::match(I, P);
    if (r) {
      std::map<Instruction *, const char *> rv;
      rv[LA] = "{ S[i] -> A[i] }";
      rv[LB] = "{ S[i] -> B[0] }";
      rv[LC] = "{ S[i] -> C[i] }";
      rv[SC] = "{ S[i] -> C[i] }";
      return rv;
    }
    else {
      return std::nullopt;
    }
  }

  virtual bool enableTiling() override { return true; }
};

class AxpyNStmtPattern : public AxpyStmtPattern{
public:
  AxpyNStmtPattern(const Dependences &D)
    : AxpyStmtPattern(D) {}

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new AxpyNEmitter(Stmt, LC, LA, LB);
  }

  virtual bool enableTiling() override { return false; }
};

class MatMulAMXStmtPattern : public MatMulStmtPattern {
public:
  MatMulAMXStmtPattern(const Dependences &D) : MatMulStmtPattern(D) {}

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new MatMulAMXEmitter(Stmt, LC, LA, LB, strideA, strideB);
  }

  virtual std::vector<unsigned> getDimensions() override { return {AMX_L, AMX_M, AMX_N}; }

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    Value *A;
    Value *B;
    Value *C;
    Instruction *SC;
    auto P = m_Store(m_Add(m_Load(m_Value(C), LC), m_Mul(m_Load(m_Value(B), LB), m_Load(m_Value(A), LA))), m_Deferred(C), SC);
    bool r = PatternMatch::match(I, P);
    if (r) {
      std::map<Instruction *, const char *> rv;
      rv[LA] = "{ S[i,j,k] -> A[i,k] }";
      rv[LB] = "{ S[i,j,k] -> B[k,j] }";
      rv[LC] = "{ S[i,j,k] -> C[i,j] }";
      rv[SC] = "{ S[i,j,k] -> C[i,j] }";
      return rv;
    }
    else {
      return std::nullopt;
    }
  }
};

class MatMulSMEStmtPattern : public MatMulStmtPattern {
public:
  MatMulSMEStmtPattern(const Dependences &D) : MatMulStmtPattern(D) {}

  virtual ReplacementEmitter *getEmitter(ScopStmt &Stmt) override {
    return new MatMulSMEEmitter(Stmt, LC, LA, LB, strideA, strideB);
  }

  virtual std::vector<unsigned> getDimensions() override { return {SME_L, SME_M, SME_N}; }

  std::optional<std::map<Instruction *, const char *>> matchSingleInstruction(Instruction *I) override {
    Value *A;
    Value *B;
    Value *C;

    Instruction *SC;

    auto P = m_Store(m_c_FAdd(m_c_FMul(m_Load(m_Value(A), LA), m_Load(m_Value(B), LB)), m_Load(m_Value(C), LC)), m_Deferred(C), SC);
    bool r = PatternMatch::match(I, P);
    if (r) {
      std::map<Instruction *, const char *> rv;
      rv[LA] = "{ S[i,j,k] -> A[k,i] }"; // Transposed matrix!
      rv[LB] = "{ S[i,j,k] -> B[k,j] }";
      rv[LC] = "{ S[i,j,k] -> C[i,j] }";
      rv[SC] = "{ S[i,j,k] -> C[i,j] }";
      return rv;
    }
    else {
      return std::nullopt;
    }
  }
};

} // namespace

char ScopMatcher::ID = 0;

void ScopMatcher::printScop(raw_ostream &OS, Scop &S) const {
  OS << S;
}

bool ScopMatcher::runOnScop(Scop &S) {
  LLVM_DEBUG(dbgs() << "runOnScop begin\n");

  if (ScopMatcherSelected == SCOP_MATCHER_NONE)
    return true;

  const Dependences &D =
      getAnalysis<DependenceInfo>().getDependences(Dependences::AL_Statement);

  int changed = 0;
  for (ScopStmt &Stmt : S) {
    if (ScopMatcherSelected == SCOP_MATCHER_DOTPROD) {
      DotProdStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_DOTPROD_INTRINSIC) {
      DotProdIntrinsicStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_DOTPROD_N) {
      DotProdNStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_AXPY) {
      AxpyStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    } if (ScopMatcherSelected == SCOP_MATCHER_AXPY_N) {
      AxpyNStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_VECT_MATMUL) {
      // Use the generated patterns here
      if (  VectMatMulStmtPattern_0_0(D).match(Stmt)
         || VectMatMulStmtPattern_1_0(D).match(Stmt)
         || VectMatMulStmtPattern_2_0(D).match(Stmt)
         || VectMatMulStmtPattern_3_0(D).match(Stmt)
         || VectMatMulStmtPattern_0_1(D).match(Stmt)
         || VectMatMulStmtPattern_1_1(D).match(Stmt)
         || VectMatMulStmtPattern_2_1(D).match(Stmt)
         || VectMatMulStmtPattern_3_1(D).match(Stmt)
         ) {
        changed++;
      }
    }
    if (ScopMatcherSelected == SCOP_MATCHER_VECT_MATMUL_MN) {
      VectMatMulMNStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
      // VectMatMulMNCommStmtPattern spc(D);
      // changed += spc.match(Stmt) ? 1 : 0;
      // VectMatMulMNBetaStmtPattern bsp(D);
      // changed += bsp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_MATMUL) {
      MatMulStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_MATMUL_LMN) {
      MatMulLMNStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_MATMUL_ALPHA_LMN) {
      MatMulAlphaLMNStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
      MatMulAlphaInterchangedLMNStmtPattern mmlmnic(D);
      changed += mmlmnic.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_EXP) {
      MatMulAlphaStmtPattern mmasp(D);
      bool matmulAlphaSuccess = mmasp.match(Stmt) ? 1 : 0;
      if (matmulAlphaSuccess) {
        changed++;
      }
      else {
        VectMatMulBetaStmtPattern bsp(D);
        changed += bsp.match(Stmt) ? 1 : 0;
      }

      MatMulAlphaInterchangedStmtPattern mmlmnic(D);
      changed += mmlmnic.match(Stmt) ? 1 : 0;

      MatMulStmtPattern mmsp(D);
      int matmulSuccess = mmsp.match(Stmt) ? 1 : 0;
      if (matmulSuccess) {
        changed++;
      }
      else {
        VectMatMulStmtPattern vmmsp(D);
        changed += vmmsp.match(Stmt) ? 1 : 0;
        VectMatMulCommStmtPattern spc(D);
        changed += spc.match(Stmt) ? 1 : 0;
      }
      AxpyStmtPattern asp(D);
      changed += asp.match(Stmt) ? 1 : 0;
      DotProdStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_EXP_N) {
      MatMulAlphaLMNStmtPattern mmasp(D);
      bool matmulAlphaSuccess = mmasp.match(Stmt) ? 1 : 0;
      if (matmulAlphaSuccess) {
        changed++;
      }
      else {
        VectMatMulMNBetaStmtPattern bsp(D);
        changed += bsp.match(Stmt) ? 1 : 0;
      }

      MatMulAlphaInterchangedLMNStmtPattern mmlmnic(D);
      changed += mmlmnic.match(Stmt) ? 1 : 0;

      MatMulLMNStmtPattern mmsp(D);
      int matmulSuccess = mmsp.match(Stmt) ? 1 : 0;
      if (matmulSuccess) {
        changed++;
      }
      else {
        VectMatMulMNStmtPattern vmmsp(D);
        changed += vmmsp.match(Stmt) ? 1 : 0;
        VectMatMulMNCommStmtPattern spc(D);
        changed += spc.match(Stmt) ? 1 : 0;
      }
      AxpyNStmtPattern asp(D);
      changed += asp.match(Stmt) ? 1 : 0;
      DotProdNStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_MATMUL_AMX) {
      MatMulAMXStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
    if (ScopMatcherSelected == SCOP_MATCHER_MATMUL_SME) {
      MatMulSMEStmtPattern sp(D);
      changed += sp.match(Stmt) ? 1 : 0;
    }
  }

  if (changed)
    LLVM_DEBUG(dbgs() << "ScopMatcher got SUCCESSFUL match " << changed << "\n");
  else
    LLVM_DEBUG(dbgs() << "ScopMatcher got NO match\n");

  LLVM_DEBUG(dbgs() << "runOnScop end\n");
  return ! changed;
}

void ScopMatcher::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();
  AU.addPreserved<DependenceInfo>();
}

Pass *polly::createScopMatcherPass() { return new ScopMatcher(); }

PreservedAnalyses ScopMatcherPass::run(Scop &S, ScopAnalysisManager &SAM,
                                       ScopStandardAnalysisResults &SAR,
                                       SPMUpdater &) {
  // This invalidates all analyses on Scop.
  PreservedAnalyses PA;
  PA.preserveSet<AllAnalysesOn<Module>>();
  PA.preserveSet<AllAnalysesOn<Function>>();
  PA.preserveSet<AllAnalysesOn<Loop>>();
  return PA;
}

INITIALIZE_PASS_BEGIN(ScopMatcher, "polly-scop-matcher",
                      "Polly - Match statements"
                      " in a Scop)",
                      false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo)
INITIALIZE_PASS_END(ScopMatcher, "polly-scop-matcher",
                    "Polly - Match statements"
                    " in a Scop)",
                    false, false);

//===----------------------------------------------------------------------===//

namespace {
/// Print result from ScopMatcher.
class ScopMatcherPrinterLegacyPass final : public ScopPass {
public:
  static char ID;

  ScopMatcherPrinterLegacyPass() : ScopMatcherPrinterLegacyPass(outs()){};
  explicit ScopMatcherPrinterLegacyPass(llvm::raw_ostream &OS)
      : ScopPass(ID), OS(OS) {}

  bool runOnScop(Scop &S) override {
    ScopMatcher &P = getAnalysis<ScopMatcher>();

    OS << "Printing analysis '" << P.getPassName() << "' for region: '"
       << S.getRegion().getNameStr() << "' in function '"
       << S.getFunction().getName() << "':\n";
    P.printScop(OS, S);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ScopPass::getAnalysisUsage(AU);
    AU.addRequired<ScopMatcher>();
    AU.setPreservesAll();
  }

private:
  llvm::raw_ostream &OS;
};

char ScopMatcherPrinterLegacyPass::ID = 0;
} // namespace

Pass *polly::createScopMatcherPrinterLegacyPass(llvm::raw_ostream &OS) {
  return new ScopMatcherPrinterLegacyPass(OS);
}

INITIALIZE_PASS_BEGIN(ScopMatcherPrinterLegacyPass, "polly-print-scop-matcher",
                      "Polly - Print Scop Matcher result", false, false)
INITIALIZE_PASS_DEPENDENCY(ScopMatcher)
INITIALIZE_PASS_END(ScopMatcherPrinterLegacyPass, "polly-print-scop-matcher",
                      "Polly - Print Scop Matcher result", false, false)

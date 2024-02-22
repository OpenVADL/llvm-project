//===-- ScopMatcherPatterns.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class IPat;

class Placeholder : IPat;

def A : Placeholder;
def B : Placeholder;
def C : Placeholder;
def D : Placeholder;
def E : Placeholder;
def F : Placeholder;

class Empty : Placeholder;
def empty : Empty;

class Stride : Placeholder;
def strideA : Stride;
def strideB : Stride;
def strideC : Stride;
def strideD : Stride;
def strideE : Stride;
def strideF : Stride;

class FAdd<IPat lp, IPat rp> : IPat {
  IPat l = lp;
  IPat r = rp;
}

class FMul<IPat lp, IPat rp> : IPat {
  IPat l = lp;
  IPat r = rp;
}

class Store<IPat v, Placeholder a> : IPat {
  Placeholder addr = a;
  IPat val = v;
}

class Load<Placeholder a> : IPat {
  Placeholder addr = a;
}

class Expression;
class Dimension : Expression;

// Per convention, dimensions start with j.
def j : Expression;
def k : Expression;
def l : Expression;
def m : Expression;
def n : Expression;

class Access<Placeholder p, list<Expression> d> {
  Placeholder placeholder = p;
  list<Expression> dims = d;
}

class ReadAccess<Placeholder p, list<Expression> d> : Access<p, d>;
class WriteAccess<Placeholder p, list<Expression> d> : Access<p, d>;


class Interchange;

class Leaf : Interchange;

class Seq<list<Interchange> l> : Interchange {
  list<Interchange> i = l;
}

class Perm<list<Interchange> l> : Interchange {
  list<Interchange> i = l;
}

def L : Leaf;


class Mode;
class Parameterized<int d> : Mode {
  int numDims = d;
}
class Fixed<list<int> d> : Mode {
  list<int> fixedDims = d;
}


class Emitter<string f, list<Placeholder> a> {
  string function = f;
  list<Placeholder> args = a;
}

class ScopMatcherPattern<IPat i, Mode m, list<Access> a, Emitter e, Interchange inter> {
  IPat iPat = i;
  Mode mode = m;
  list<Access> accesses = a;
  Emitter emitter = e;
  Interchange interchange = inter;
}


def InterchangeFoo : Seq<[L, Perm<[L, L]>]>;
def InterchangeBar : Perm<[L, L, L]>;



def MACPat :
  Store<
    FAdd<
      FMul<Load<A>, Load<B>>,
      Load<C>>,
    C>;

def LA : ReadAccess<A,  [k]>;
def LB : ReadAccess<B,  [k, j]>;
def LC : ReadAccess<C,  [j]>;
def SC : WriteAccess<C, [j]>;

def VectMatMulStmtPattern :
  ScopMatcherPattern< MACPat
                    , Fixed<[64, 64]>
                    , [LA, LB, LC, SC]
                    // , Emitter<"vect_matmul_fixed", [C, A, B, strideB]>
                    , Emitter<"VectMatMulEmitter", [C, A, B, empty, strideB]>
                    , Perm<[L, L]>>;
                    // , Seq<[L]>>;
                    // , L>;
                    // , Seq<[ L, Perm<[L, L, L, L]>]>  >;
                    // , Seq<[ L, L, Perm<[L, L, L]>, L]>  >;

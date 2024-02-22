//===- polly/ScopMatcher.h -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOPMATCHER_H
#define POLLY_SCOPMATCHER_H

#include "polly/ScopPass.h"
#include "polly/CodeGen/IRBuilder.h"
#include "llvm/IR/PassManager.h"

namespace polly {
llvm::Pass *createScopMatcherPass();
llvm::Pass *createScopMatcherPrinterLegacyPass(llvm::raw_ostream &OS);


struct ScopMatcherPass final : llvm::PassInfoMixin<ScopMatcherPass> {
  llvm::PreservedAnalyses run(Scop &, ScopAnalysisManager &,
                              ScopStandardAnalysisResults &, SPMUpdater &);
};

class ReplacementEmitter {
public:
  ReplacementEmitter(ScopStmt &Stmt) : Stmt(Stmt) {}
  virtual ~ReplacementEmitter() {}
  virtual void addUpperBound(Value *ub) {};

  ScopStmt &Stmt;

  using GenerateLocation = std::function<Value*(Instruction *)>;
  virtual void emit(PollyIRBuilder &Builder,
                    const DebugLoc &Loc,
                    GenerateLocation genLoc) = 0;
};

} // namespace polly

namespace llvm {
void initializeScopMatcherPass(llvm::PassRegistry &);
void initializeScopMatcherPrinterLegacyPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_SCOPMATCHER_H */

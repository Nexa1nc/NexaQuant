#pragma once
#include <vector>
#include <string>

struct LlmModel {
    std::string name;
};

class SpeculativeEngine {
private:
    int draft_lookahead;
public:
    SpeculativeEngine(int lookahead = 4) : draft_lookahead(lookahead) {}
    
    void run_step() {
        std::cout << "[Speculative] Generating with Draft, verifying with Target...\n";
    }
};

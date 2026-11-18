#pragma once

#include <map>
#include <stdexcept>
#include <string>

namespace mg
{
    using Variables = std::map<std::string, double>;

    struct UnknownSymbol : std::runtime_error
    {
        explicit UnknownSymbol(const std::string &name)
            : std::runtime_error("unknown variable '" + name + "'"), symbol(name) {}
        std::string symbol;
    };

    // Evaluates arithmetic such as "$d_model * 4", "d_model / num_heads" or
    // "(a + 1) // 2". Supports + - * / // % and parentheses; variable names may
    // be written with or without a leading '$'.
    double Evaluate(const std::string &expression, const Variables &variables);

    bool IsIdentifier(const std::string &text);
}

#include "Graph/Expr.h"

#include <cctype>
#include <cmath>

namespace mg
{
    namespace
    {
        class Parser
        {
        public:
            Parser(const std::string &text, const Variables &vars) : m_text(text), m_vars(vars) {}

            double Parse()
            {
                const double value = Expr();
                Skip();
                if (m_pos != m_text.size())
                {
                    Fail("unexpected '" + std::string(1, m_text[m_pos]) + "'");
                }
                return value;
            }

        private:
            double Expr()
            {
                double value = Term();
                for (;;)
                {
                    if (Accept("+")) value += Term();
                    else if (Accept("-")) value -= Term();
                    else return value;
                }
            }

            double Term()
            {
                double value = Factor();
                for (;;)
                {
                    if (Accept("//"))
                    {
                        const double rhs = Factor();
                        if (rhs == 0) Fail("division by zero");
                        value = std::floor(value / rhs);
                    }
                    else if (Accept("*")) value *= Factor();
                    else if (Accept("/"))
                    {
                        const double rhs = Factor();
                        if (rhs == 0) Fail("division by zero");
                        value /= rhs;
                    }
                    else if (Accept("%"))
                    {
                        const double rhs = Factor();
                        if (rhs == 0) Fail("division by zero");
                        value = std::fmod(value, rhs);
                    }
                    else return value;
                }
            }

            double Factor()
            {
                if (Accept("-")) return -Factor();
                if (Accept("+")) return Factor();
                return Primary();
            }

            double Primary()
            {
                Skip();
                if (Accept("("))
                {
                    const double value = Expr();
                    if (!Accept(")")) Fail("missing ')'");
                    return value;
                }
                if (m_pos < m_text.size() && (std::isdigit(static_cast<unsigned char>(m_text[m_pos])) || m_text[m_pos] == '.'))
                {
                    size_t used = 0;
                    const double value = std::stod(m_text.substr(m_pos), &used);
                    m_pos += used;
                    return value;
                }
                if (m_pos < m_text.size() && m_text[m_pos] == '$') ++m_pos;
                const size_t start = m_pos;
                while (m_pos < m_text.size() && (std::isalnum(static_cast<unsigned char>(m_text[m_pos])) || m_text[m_pos] == '_'))
                {
                    ++m_pos;
                }
                if (start == m_pos) Fail("expected a number or variable");
                const std::string name = m_text.substr(start, m_pos - start);
                const auto it = m_vars.find(name);
                if (it == m_vars.end()) throw UnknownSymbol(name);
                return it->second;
            }

            bool Accept(const char *token)
            {
                Skip();
                const std::string t(token);
                if (m_text.compare(m_pos, t.size(), t) == 0)
                {
                    // "/" must not swallow the first half of "//".
                    if (t == "/" && m_text.compare(m_pos, 2, "//") == 0) return false;
                    m_pos += t.size();
                    return true;
                }
                return false;
            }

            void Skip()
            {
                while (m_pos < m_text.size() && std::isspace(static_cast<unsigned char>(m_text[m_pos]))) ++m_pos;
            }

            [[noreturn]] void Fail(const std::string &message) const
            {
                throw std::invalid_argument("expression '" + m_text + "': " + message);
            }

            const std::string &m_text;
            const Variables &m_vars;
            size_t m_pos = 0;
        };
    }

    double Evaluate(const std::string &expression, const Variables &variables)
    {
        return Parser(expression, variables).Parse();
    }

    bool IsIdentifier(const std::string &text)
    {
        if (text.empty() || std::isdigit(static_cast<unsigned char>(text[0]))) return false;
        for (const char c : text)
        {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') return false;
        }
        return true;
    }
}

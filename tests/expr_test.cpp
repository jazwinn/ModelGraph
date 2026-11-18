#include <gtest/gtest.h>

#include "Graph/Expr.h"
#include "Graph/Registry.h"

using namespace mg;

TEST(Expr, Arithmetic)
{
    const Variables v = {{"d_model", 768}, {"num_heads", 12}};
    EXPECT_DOUBLE_EQ(Evaluate("$d_model * 4", v), 3072);
    EXPECT_DOUBLE_EQ(Evaluate("d_model / num_heads", v), 64);
    EXPECT_DOUBLE_EQ(Evaluate("(1 + 2) * -3", v), -9);
    EXPECT_DOUBLE_EQ(Evaluate("7 // 2", v), 3);
    EXPECT_DOUBLE_EQ(Evaluate("7 % 4", v), 3);
    EXPECT_DOUBLE_EQ(Evaluate("1e-5", v), 1e-5);
}

TEST(Expr, Errors)
{
    EXPECT_THROW(Evaluate("$nope", {}), UnknownSymbol);
    EXPECT_THROW(Evaluate("1 +", {}), std::invalid_argument);
    EXPECT_THROW(Evaluate("4 / 0", {}), std::invalid_argument);
    EXPECT_THROW(Evaluate("(1", {}), std::invalid_argument);
}

TEST(Expr, ShapesMixSymbolsAndExpressions)
{
    const auto dims = ParseShape("B, S, $d_model * 2", {{"d_model", 8}});
    ASSERT_EQ(dims.size(), 3u);
    EXPECT_EQ(dims[0].symbol, "B");
    EXPECT_EQ(dims[1].symbol, "S");
    EXPECT_EQ(dims[2].value, 16);
    EXPECT_THROW(ParseShape("B, $missing", {}), GraphError);
}

#include <gtest/gtest.h>

#include "Graph/Document.h"
#include "Graph/Templates.h"

using namespace mg;

TEST(Document, JsonRoundTrip)
{
    const Document a = MakeTemplate("gpt2");
    const Document b = Document::FromJson(a.ToJson());
    EXPECT_EQ(a.ToJson(), b.ToJson());
    EXPECT_EQ(b.subgraphs.size(), 3u);   // Block, Attention, MLP
    EXPECT_EQ(b.model.format, "gpt2");
}

TEST(Document, ConnectReplacesExistingInputLink)
{
    Graph g;
    const int a = g.Add("TokenIds", "a").id;
    const int b = g.Add("TokenIds", "b").id;
    const int e = g.Add("Embedding", "e").id;
    g.Connect(a, "ids", e, "ids");
    g.Connect(b, "ids", e, "ids");
    ASSERT_EQ(g.links.size(), 1u);
    EXPECT_EQ(g.links[0].from.node, b);
}

TEST(Document, UniqueNames)
{
    Graph g;
    g.Add("Add", "add");
    EXPECT_EQ(g.Add("Add", "add").name, "add_1");
    EXPECT_EQ(g.Add("Add", "add").name, "add_2");
}

TEST(Document, VariablesCanReferenceEachOther)
{
    Document d;
    d.variables["d_model"] = int64_t{64};
    d.variables["num_heads"] = int64_t{4};
    d.variables["head_dim"] = std::string("$d_model / $num_heads");
    EXPECT_DOUBLE_EQ(d.NumericVariables().at("head_dim"), 16);
}

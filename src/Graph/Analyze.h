#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "Graph/Registry.h"

namespace mg
{
    // Static analysis of one graph: execution order, the type of every output
    // port, per-node errors and parameter counts. Runs on every edit in the
    // editor, and is the first step of compilation.
    struct Analysis
    {
        std::vector<int> order;                              // node ids, topological
        std::map<int, std::vector<TensorType>> outputs;      // node id -> output types
        std::map<int, std::vector<std::optional<TensorType>>> inputs;  // node id -> input types by port
        std::map<int, std::string> errors;                   // node id -> message
        std::map<int, int64_t> params;                       // node id -> parameter count
        std::map<int, NodePorts> ports;                      // node id -> resolved ports
        std::vector<TensorType> results;                     // types reaching Output nodes, by index
        std::vector<std::string> graphErrors;                // problems not tied to one node
        int64_t totalParams = 0;

        bool Ok() const { return errors.empty() && graphErrors.empty(); }
        std::string Summary() const;
    };

    // `inputs` overrides the declared shapes of the graph's Input nodes (used
    // when a subgraph is analysed as part of a Repeat).
    Analysis Analyze(const Document &doc, const Graph &graph, const Registry &registry,
                     const std::optional<std::vector<TensorType>> &inputs = std::nullopt, int depth = 0);


}

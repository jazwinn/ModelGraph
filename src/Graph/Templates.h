#pragma once

#include <string>
#include <vector>

#include "Graph/Document.h"
#include "NeuralNet/Transformer/TransformerConfig.h"

namespace mg
{
    // Builds the graph equivalent of InferenceEngine's models::CausalModel for a
    // config. Parameter names match CausalModel exactly, so its checkpoint
    // loaders work unchanged and the two can be compared output-for-output.
    Document FromTransformerConfig(const nn::transformer::config::TransformerConfig &cfg, const std::string &name);

    // Named starting points for "File > New from template".
    struct TemplateInfo
    {
        std::string id;
        std::string title;
    };
    std::vector<TemplateInfo> Templates();
    Document MakeTemplate(const std::string &id);
}

#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "Graph/Document.h"
#include "Models/WeightLoader.h"

namespace mg
{
    struct LoadReport
    {
        size_t loaded = 0;
        std::vector<std::string> missing;      // parameters the checkpoint did not provide
        std::vector<std::string> unused;       // checkpoint tensors nothing asked for
        std::vector<std::string> mismatched;   // name matched, shape did not
        std::set<std::string> loadedNames;

        std::string Text(size_t limit = 8) const;
    };

    // Reads a checkpoint and maps its tensor names onto the graph's parameter
    // names: "gpt2" and "llama" reuse InferenceEngine's converters, "raw"
    // applies only the document's own rename rules.
    models::Weights<Scalar> ReadCheckpoint(const Document &doc);

    // Unlike models::LoadWeights this never throws on a partial match: when
    // you are experimenting, a changed layer simply starts from fresh weights.
    LoadReport LoadLenient(nn::Module<Scalar> &model, models::Weights<Scalar> weights);

    // Gives every parameter not in `keep` a fresh initialisation: N(0, 0.02)
    // for matrices; vectors keep their constructed value (norm 1, bias 0).
    void InitializeFresh(nn::Module<Scalar> &model, const std::set<std::string> &keep, uint64_t seed = 1234);

    // Copies parameters with matching name and shape from an old build, so
    // editing one node keeps every other node's weights. Returns copied names.
    std::set<std::string> CarryOver(const nn::Module<Scalar> &from, nn::Module<Scalar> &to);
}

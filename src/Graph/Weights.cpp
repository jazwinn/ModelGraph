#include "Graph/Weights.h"

#include <map>
#include <random>

#include "Models/ModelConfigs.h"
#include "Models/gpt2/Checkpoint.h"
#include "Serialize/Safetensors.h"

namespace mg
{
    std::string LoadReport::Text(size_t limit) const
    {
        auto list = [limit](const std::vector<std::string> &names)
        {
            std::string s;
            for (size_t i = 0; i < names.size() && i < limit; ++i) s += "\n    " + names[i];
            if (names.size() > limit) s += "\n    ... and " + std::to_string(names.size() - limit) + " more";
            return s;
        };
        std::string text = "loaded " + std::to_string(loaded) + " tensors";
        if (!missing.empty()) text += "\n  fresh (not in checkpoint): " + std::to_string(missing.size()) + list(missing);
        if (!mismatched.empty()) text += "\n  shape mismatch (fresh): " + std::to_string(mismatched.size()) + list(mismatched);
        if (!unused.empty()) text += "\n  unused checkpoint tensors: " + std::to_string(unused.size()) + list(unused);
        return text;
    }

    models::Weights<Scalar> ReadCheckpoint(const Document &doc)
    {
        const std::string path = doc.Resolve(doc.model.checkpoint).string();
        if (doc.model.checkpoint.empty()) throw GraphError("no checkpoint is set");

        std::vector<models::RenameRule> rules;
        for (const RenameRuleSpec &r : doc.model.rules) rules.push_back({r.from, r.to});

        models::Weights<Scalar> weights;
        if (doc.model.format == "gpt2")
        {
            weights = models::gpt2::ReadCheckpoint<Scalar>(path);
        }
        else if (doc.model.format == "llama")
        {
            weights = models::Rename(serialize::ReadSafetensors<Scalar>(path), models::llama::Rules());
        }
        else if (doc.model.format == "raw")
        {
            weights = serialize::ReadSafetensors<Scalar>(path);
        }
        else
        {
            throw GraphError("unknown checkpoint format '" + doc.model.format + "' (use raw, gpt2 or llama)");
        }
        // Document rules run last so they can patch names on top of a preset.
        return rules.empty() ? std::move(weights) : models::Rename(std::move(weights), rules);
    }

    LoadReport LoadLenient(nn::Module<Scalar> &model, models::Weights<Scalar> weights)
    {
        LoadReport report;
        for (const auto &[name, param] : model.NamedParameters())
        {
            auto it = weights.find(name);
            if (it == weights.end())
            {
                report.missing.push_back(name);
                continue;
            }
            if (it->second.Shape() != param->Shape())
            {
                report.mismatched.push_back(name + " (checkpoint " + models::detail::ShapeText(it->second.Shape()) +
                                            ", graph " + models::detail::ShapeText(param->Shape()) + ")");
                weights.erase(it);
                continue;
            }
            *param = std::move(it->second).To(param->GetDevice());
            weights.erase(it);
            report.loadedNames.insert(name);
            ++report.loaded;
        }
        for (const auto &[name, t] : weights) report.unused.push_back(name);
        return report;
    }

    void InitializeFresh(nn::Module<Scalar> &model, const std::set<std::string> &keep, uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> normal(0.0f, 0.02f);
        for (const auto &[name, param] : model.NamedParameters())
        {
            if (keep.contains(name) || param->Rank() < 2) continue;
            FTensor host(param->Shape());
            Scalar *data = host.Data();
            for (size_t i = 0; i < host.Size(); ++i) data[i] = normal(rng);
            *param = host.To(param->GetDevice());
        }
    }

    std::set<std::string> CarryOver(const nn::Module<Scalar> &from, nn::Module<Scalar> &to)
    {
        std::map<std::string, std::shared_ptr<FTensor>> old;
        for (const auto &[name, p] : from.NamedParameters()) old[name] = p;
        std::set<std::string> copied;
        for (const auto &[name, p] : to.NamedParameters())
        {
            const auto it = old.find(name);
            if (it == old.end() || it->second->Shape() != p->Shape()) continue;
            *p = it->second->To(p->GetDevice());
            copied.insert(name);
        }
        return copied;
    }
}

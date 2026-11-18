#include "Graph/Templates.h"

#include <cstdio>

#include "Models/ModelConfigs.h"

namespace mg
{
    namespace cfgns = nn::transformer::config;

    namespace
    {
        std::string ActivationName(tensor::Activation a)
        {
            switch (a)
            {
            case tensor::Activation::GELU: return "gelu";
            case tensor::Activation::GELUTanh: return "gelu_tanh";
            case tensor::Activation::SiLU: return "silu";
            case tensor::Activation::ReLU: return "relu";
            case tensor::Activation::Sigmoid: return "sigmoid";
            case tensor::Activation::Tanh: return "tanh";
            }
            return "gelu";
        }

        std::string ScalingName(cfgns::RopeScalingType t)
        {
            switch (t)
            {
            case cfgns::RopeScalingType::Linear: return "linear";
            case cfgns::RopeScalingType::Llama3: return "llama3";
            default: return "default";
            }
        }

        AttrValue S(const char *expr) { return std::string(expr); }

        // 1e-5f widens to 9.99999974e-06; keep the value a person typed.
        double Tidy(float v)
        {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.6g", static_cast<double>(v));
            return std::stod(buf);
        }

        AttrMap RopeAttrs(const cfgns::TransformerConfig &cfg)
        {
            return {{"rope_theta", S("$rope_theta")}, {"max_seq_len", S("$max_seq_len")},
                    {"rope_scaling", ScalingName(cfg.rope_scaling.type)}, {"rope_factor", cfg.rope_scaling.factor},
                    {"rope_low_freq_factor", cfg.rope_scaling.low_freq_factor},
                    {"rope_high_freq_factor", cfg.rope_scaling.high_freq_factor},
                    {"rope_original_max_position", static_cast<int64_t>(cfg.rope_scaling.original_max_position)}};
        }

        AttrMap InputAttrs() { return {{"index", int64_t{0}}, {"dtype", std::string("f32")}, {"shape", S("B, S, $d_model")}}; }

        // Self-attention from primitives. Node names match CausalModel's
        // parameter names (attention.query.weight, ...).
        Graph MakeAttention(const cfgns::TransformerConfig &cfg)
        {
            Graph g;
            const bool rope = cfg.position == cfgns::PositionEncoding::Rotary;
            const int x = g.Add("Input", "x", InputAttrs(), 0, 120).id;
            const int q = g.Add("Linear", "query", {{"in_features", S("$d_model")}, {"out_features", S("$num_heads * $head_dim")}, {"bias", cfg.attention_bias}}, 200, 0).id;
            const int k = g.Add("Linear", "key", {{"in_features", S("$d_model")}, {"out_features", S("$num_kv_heads * $head_dim")}, {"bias", cfg.attention_bias}}, 200, 120).id;
            const int v = g.Add("Linear", "value", {{"in_features", S("$d_model")}, {"out_features", S("$num_kv_heads * $head_dim")}, {"bias", cfg.attention_bias}}, 200, 240).id;
            const int qh = g.Add("SplitHeads", "q_heads", {{"num_heads", S("$num_heads")}, {"head_dim", S("$head_dim")}}, 420, 0).id;
            const int kh = g.Add("SplitHeads", "k_heads", {{"num_heads", S("$num_kv_heads")}, {"head_dim", S("$head_dim")}}, 420, 120).id;
            const int vh = g.Add("SplitHeads", "v_heads", {{"num_heads", S("$num_kv_heads")}, {"head_dim", S("$head_dim")}}, 420, 240).id;
            for (int lin : {q, k, v}) g.Connect(x, "out", lin, "x");
            g.Connect(q, "y", qh, "x");
            g.Connect(k, "y", kh, "x");
            g.Connect(v, "y", vh, "x");
            int qOut = qh, kOut = kh;
            std::string qPort = "heads", kPort = "heads";
            if (rope)
            {
                qOut = g.Add("RoPE", "rope_q", RopeAttrs(cfg), 640, 0).id;
                kOut = g.Add("RoPE", "rope_k", RopeAttrs(cfg), 640, 120).id;
                g.Connect(qh, "heads", qOut, "x");
                g.Connect(kh, "heads", kOut, "x");
                qPort = kPort = "y";
            }
            const int cache = g.Add("KVCache", "cache", {}, 860, 180).id;
            const int sdpa = g.Add("SDPA", "sdpa", {{"causal", cfg.causal}}, 1080, 60).id;
            const int merge = g.Add("MergeHeads", "merge", {}, 1300, 60).id;
            const int o = g.Add("Linear", "output_proj", {{"in_features", S("$num_heads * $head_dim")}, {"out_features", S("$d_model")}, {"bias", cfg.output_bias}}, 1520, 60).id;
            const int y = g.Add("Output", "y", {{"index", int64_t{0}}}, 1740, 60).id;
            g.Connect(kOut, kPort, cache, "k");
            g.Connect(vh, "heads", cache, "v");
            g.Connect(qOut, qPort, sdpa, "q");
            g.Connect(cache, "k_all", sdpa, "k");
            g.Connect(cache, "v_all", sdpa, "v");
            g.Connect(sdpa, "out", merge, "heads");
            g.Connect(merge, "x", o, "x");
            g.Connect(o, "y", y, "value");
            return g;
        }

        // Feed-forward block from primitives: down(act(up(x))) or, gated,
        // down(act(gate(x)) * up(x)).
        Graph MakeMlp(const cfgns::TransformerConfig &cfg)
        {
            Graph g;
            const bool gated = cfg.mlp_type == cfgns::MLPType::Gated;
            const AttrMap upAttrs = {{"in_features", S("$d_model")}, {"out_features", S("$ffn_dim")}, {"bias", cfg.output_bias}};
            const int x = g.Add("Input", "x", InputAttrs(), 0, 60).id;
            const int up = g.Add("Linear", "up", upAttrs, 220, gated ? 120.0f : 60.0f).id;
            const int act = g.Add("Activation", "act", {{"kind", ActivationName(cfg.activation)}}, 440, 0).id;
            const int down = g.Add("Linear", "down", {{"in_features", S("$ffn_dim")}, {"out_features", S("$d_model")}, {"bias", cfg.output_bias}}, 880, 60).id;
            const int y = g.Add("Output", "y", {{"index", int64_t{0}}}, 1100, 60).id;
            g.Connect(x, "out", up, "x");
            if (gated)
            {
                const int gate = g.Add("Linear", "gate", upAttrs, 220, 0).id;
                const int mul = g.Add("Mul", "gated", {}, 660, 60).id;
                g.Connect(x, "out", gate, "x");
                g.Connect(gate, "y", act, "x");
                g.Connect(act, "y", mul, "a");
                g.Connect(up, "y", mul, "b");
                g.Connect(mul, "out", down, "x");
            }
            else
            {
                g.Connect(up, "y", act, "x");
                g.Connect(act, "y", down, "x");
            }
            g.Connect(down, "y", y, "value");
            return g;
        }

        Graph MakeBlock(const cfgns::TransformerConfig &cfg)
        {
            Graph g;
            const bool pre = cfg.norm_placement == cfgns::NormPlacement::Pre;
            const std::string norm = cfg.norm_type == cfgns::NormType::RMSNorm ? "RMSNorm" : "LayerNorm";
            const AttrMap normAttrs = {{"dim", S("$d_model")}, {"eps", S("$norm_eps")}};
            const AttrMap attn = {{"subgraph", std::string("Attention")}};
            const AttrMap mlp = {{"subgraph", std::string("MLP")}};

            const int in = g.Add("Input", "x", InputAttrs(), 0, 0).id;
            if (pre)
            {
                const int n1 = g.Add(norm, "attenNorm", normAttrs, 220, 0).id;
                const int at = g.Add("Group", "attention", attn, 440, 0).id;
                const int r1 = g.Add("Add", "residual_attn", {}, 660, 40).id;
                const int n2 = g.Add(norm, "mlpNorm", normAttrs, 880, 0).id;
                const int ml = g.Add("Group", "mlp", mlp, 1100, 0).id;
                const int r2 = g.Add("Add", "residual_mlp", {}, 1320, 40).id;
                const int out = g.Add("Output", "y", {{"index", int64_t{0}}}, 1540, 40).id;
                g.Connect(in, "out", n1, "x");
                g.Connect(n1, "y", at, "x");
                g.Connect(in, "out", r1, "a");
                g.Connect(at, "y", r1, "b");
                g.Connect(r1, "out", n2, "x");
                g.Connect(n2, "y", ml, "x");
                g.Connect(r1, "out", r2, "a");
                g.Connect(ml, "y", r2, "b");
                g.Connect(r2, "out", out, "value");
            }
            else
            {
                const int at = g.Add("Group", "attention", attn, 220, 0).id;
                const int r1 = g.Add("Add", "residual_attn", {}, 440, 40).id;
                const int n1 = g.Add(norm, "attenNorm", normAttrs, 660, 40).id;
                const int ml = g.Add("Group", "mlp", mlp, 880, 0).id;
                const int r2 = g.Add("Add", "residual_mlp", {}, 1100, 40).id;
                const int n2 = g.Add(norm, "mlpNorm", normAttrs, 1320, 40).id;
                const int out = g.Add("Output", "y", {{"index", int64_t{0}}}, 1540, 40).id;
                g.Connect(in, "out", at, "x");
                g.Connect(in, "out", r1, "a");
                g.Connect(at, "y", r1, "b");
                g.Connect(r1, "out", n1, "x");
                g.Connect(n1, "y", ml, "x");
                g.Connect(n1, "y", r2, "a");
                g.Connect(ml, "y", r2, "b");
                g.Connect(r2, "out", n2, "x");
                g.Connect(n2, "y", out, "value");
            }
            return g;
        }
    }

    Document FromTransformerConfig(const cfgns::TransformerConfig &cfg, const std::string &name)
    {
        cfg.Validate();
        Document d;
        d.name = name;
        d.variables = {
            {"vocab_size", static_cast<int64_t>(cfg.vocab_size)},
            {"d_model", static_cast<int64_t>(cfg.d_model)},
            {"num_layers", static_cast<int64_t>(cfg.num_layers)},
            {"num_heads", static_cast<int64_t>(cfg.num_heads)},
            {"num_kv_heads", static_cast<int64_t>(cfg.KVHeads())},
            {"head_dim", static_cast<int64_t>(cfg.HeadDim())},
            {"ffn_dim", static_cast<int64_t>(cfg.ffn_dim)},
            {"max_seq_len", static_cast<int64_t>(cfg.max_seq_len)},
            {"norm_eps", Tidy(cfg.norm_eps)},
            {"rope_theta", Tidy(cfg.rope_theta)},
        };
        d.model.eos = cfg.eos_token_ids;

        d.subgraphs["Block"] = MakeBlock(cfg);
        d.subgraphs["Attention"] = MakeAttention(cfg);
        d.subgraphs["MLP"] = MakeMlp(cfg);

        Graph &g = d.main;
        const std::string norm = cfg.norm_type == cfgns::NormType::RMSNorm ? "RMSNorm" : "LayerNorm";
        const int ids = g.Add("TokenIds", "tokens", {}, 0, 0).id;
        const int wte = g.Add("Embedding", "wte", {{"num_embeddings", S("$vocab_size")}, {"dim", S("$d_model")}}, 220, 0).id;
        g.Connect(ids, "ids", wte, "ids");
        int x = wte;
        std::string xPort = "out";

        float col = 440;
        if (cfg.position == cfgns::PositionEncoding::Learned)
        {
            const int pos = g.Add("Positions", "positions", {}, 220, 160).id;
            const int wpe = g.Add("Embedding", "wpe", {{"num_embeddings", S("$max_seq_len")}, {"dim", S("$d_model")}}, 440, 160).id;
            const int add = g.Add("Add", "embed_sum", {}, 660, 40).id;
            g.Connect(ids, "ids", pos, "ids");
            g.Connect(pos, "positions", wpe, "ids");
            g.Connect(wte, "out", add, "a");
            g.Connect(wpe, "out", add, "b");
            x = add;
            col = 880;
        }

        const int rep = g.Add("Repeat", "transformer.h", {{"subgraph", std::string("Block")}, {"count", S("$num_layers")}}, col, 40).id;
        g.Find(rep)->ui = {{"expanded", true}};   // show the layer's structure when opened
        g.Connect(x, xPort, rep, "x");
        x = rep;
        xPort = "y";
        col += 220;

        if (cfg.norm_placement == cfgns::NormPlacement::Pre)
        {
            const int fn = g.Add(norm, "transformer.finalNorm", {{"dim", S("$d_model")}, {"eps", S("$norm_eps")}}, col, 40).id;
            g.Connect(x, xPort, fn, "x");
            x = fn;
            col += 220;
        }

        int head;
        if (cfg.tie_word_embeddings)
        {
            head = g.Add("TiedLinear", "lm_head", {{"weight_from", std::string("wte")}}, col, 40).id;
        }
        else
        {
            head = g.Add("Linear", "lm_head", {{"in_features", S("$d_model")}, {"out_features", S("$vocab_size")}, {"bias", false}}, col, 40).id;
        }
        g.Connect(x, xPort, head, "x");
        const int logits = g.Add("Logits", "logits", {}, col + 220, 40).id;
        g.Connect(head, "y", logits, "logits");
        return d;
    }

    std::vector<TemplateInfo> Templates()
    {
        return {
            {"gpt2", "GPT-2 small (124M)"},
            {"llama3.2-1b", "Llama 3.2 1B"},
            {"tiny", "Tiny Llama-style (random weights)"},
        };
    }

    Document MakeTemplate(const std::string &id)
    {
        if (id == "gpt2")
        {
            Document d = FromTransformerConfig(models::gpt2::Small(), "GPT-2 small");
            d.model.format = "gpt2";
            d.model.checkpoint = "../../InferenceEngine/Models/gpt2/model.safetensors";
            d.model.tokenizer = "../../InferenceEngine/Models/gpt2/tokenizer.json";
            d.model.contextSize = 1024;
            return d;
        }
        if (id == "llama3.2-1b")
        {
            Document d = FromTransformerConfig(models::llama::Llama32_1B(), "Llama 3.2 1B Instruct");
            d.model.format = "llama";
            d.model.checkpoint = "../../InferenceEngine/Models/llama3.2-instruct/model.safetensors";
            d.model.tokenizer = "../../InferenceEngine/Models/llama3.2-instruct/tokenizer.json";
            d.model.chatTemplate = "../../InferenceEngine/Models/llama3.2-instruct/tokenizer_config.json";
            d.model.contextSize = 2048;
            return d;
        }
        if (id == "tiny")
        {
            cfgns::TransformerConfig c;
            c.vocab_size = 256; c.d_model = 64; c.num_layers = 2; c.num_heads = 4; c.num_kv_heads = 2;
            c.ffn_dim = 172; c.max_seq_len = 256;
            c.norm_type = cfgns::NormType::RMSNorm; c.mlp_type = cfgns::MLPType::Gated;
            c.activation = tensor::Activation::SiLU; c.position = cfgns::PositionEncoding::Rotary;
            c.attention_bias = false; c.output_bias = false;
            Document d = FromTransformerConfig(c, "Tiny");
            d.model.device = "cpu";
            d.model.contextSize = 256;
            return d;
        }
        throw GraphError("unknown template '" + id + "'");
    }
}

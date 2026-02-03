# DeepSeek-OCR2-query-ablation

This codebase is used to ablate the role of causal flow tokens in DeepSeek-OCR2 model. For detailed analysis, you can refer to https://github.com/sen-ye/Analysis-DeepSeek-OCR2.

## How to use
**Step1**. Download DeepSeek-OCR2 model weights

```
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
```

**Step2**. 
For encoder part ablation, please run

```
# Set WEIGHTS_DIR
bash scripts/run_encoder_query_attn.sh
```

For decoder part ablation, plase run

```
# Set WEIGHTS_DIR
bash scripts/run_decoder_ablation.sh
```

Ablation results are saved to ```./results```

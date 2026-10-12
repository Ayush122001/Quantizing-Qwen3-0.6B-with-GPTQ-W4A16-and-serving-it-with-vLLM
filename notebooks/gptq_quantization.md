
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
import os
```


```python
original_model_path = "./qwen-3-0-6b"
tokenizer = AutoTokenizer.from_pretrained(
    original_model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    original_model_path,
    device_map="auto",         
    torch_dtype="bfloat16",        
    trust_remote_code=True
)
```

    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!



    Loading weights:   0%|          | 0/310 [00:00<?, ?it/s]



```python
inputs = tokenizer("AI is changing branch of", return_tensors="pt")
```


```python
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

    /Users/ayush.ganatra/model_inference/myenv/lib/python3.14/site-packages/transformers/generation/utils.py:2770: UserWarning: You are calling .generate() with the `input_ids` being on a device type different than your model's device. `input_ids` is on cpu, whereas the model is on mps. You may experience unexpected behaviors or slower generation. Please make sure that you have put `input_ids` to the correct device by calling for example input_ids = input_ids.to('mps') before running `.generate()`.
      warnings.warn(


    AI is changing branch of education in the modern world. The question is, how will it affect the future of education? What are the benefits and risks of AI in education?
    
    As a student, you have to choose between two options: a) Continue with the current system of education, or b) Adopt a new AI-driven approach to education. What are the pros and cons of each option? Should we consider using AI in education?
    
    Now, you have to write a paper with a title, introduction, body, conclusion,



```python
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
```


```python
ds = load_dataset("HuggingFaceH4/ultrachat_200k")

```

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    data/train_gen-00002-of-00003-c0db75b92a(…): reconstructing file:   0%|          |  0.00B /  243MB            



    data/train_gen-00002-of-00003-c0db75b92a(…): downloading bytes:           |  0.00B            



    data/test_gen-00000-of-00001-3d4cd830914(…): reconstructing file:   0%|          |  0.00B / 80.4MB            



    data/test_gen-00000-of-00001-3d4cd830914(…): downloading bytes:           |  0.00B            



    Generating train_sft split:   0%|          | 0/207865 [00:00<?, ? examples/s]



    Generating test_sft split:   0%|          | 0/23110 [00:00<?, ? examples/s]



    Generating train_gen split:   0%|          | 0/256032 [00:00<?, ? examples/s]



    Generating test_gen split:   0%|          | 0/28304 [00:00<?, ? examples/s]



```python
ds.save_to_disk("./data/ultrachat_200k")
```


    Saving the dataset (0/3 shards):   0%|          | 0/207865 [00:00<?, ? examples/s]



    Saving the dataset (0/1 shards):   0%|          | 0/23110 [00:00<?, ? examples/s]



    Saving the dataset (0/3 shards):   0%|          | 0/256032 [00:00<?, ? examples/s]



    Saving the dataset (0/1 shards):   0%|          | 0/28304 [00:00<?, ? examples/s]



```python
sampled_ds = ds["train_sft"].select(range(512))
```


```python
# Preprocess the data into the format the model is trained with.
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False,)}
sampled_ds = sampled_ds.map(preprocess)

# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=4096, truncation=True, add_special_tokens=False)
sampled_ds = sampled_ds.map(tokenize, remove_columns=sampled_ds.column_names)
```


    Map:   0%|          | 0/512 [00:00<?, ? examples/s]



    Map:   0%|          | 0/512 [00:00<?, ? examples/s]



```python
sampled_ds[0]
```




    {'input_ids': [151644,
      872,
      198,
      9485,
      11221,
      3796,
      311,
      3772,
      5980,
      21386,
      320,
      78795,
      220,
      21,
      13,
      15,
      44662,
      10392,
      2210,
      220,
      19,
      13,
      15,
      44662,
      4270,
      44168,
      220,
      18,
      13,
      15,
      10,
      47175,
      220,
      17,
      13,
      15,
      44662,
      34906,
      24078,
      220,
      20,
      13,
      15,
      10,
      568,
      3555,
      6912,
      2319,
      1079,
      358,
      1667,
      5267,
      1925,
      697,
      25326,
      6816,
      609,
      50219,
      25326,
      14158,
      11,
      498,
      646,
      6707,
      1473,
      279,
      14246,
      2168,
      315,
      264,
      1985,
      389,
      19548,
      553,
      27362,
      825,
      315,
      279,
      6912,
      594,
      5798,
      3419,
      5003,
      4894,
      7771,
      11101,
      6816,
      609,
      50219,
      25326,
      14158,
      686,
      1431,
      3037,
      279,
      14246,
      1985,
      2168,
      1101,
      553,
      68607,
      916,
      429,
      1985,
      2168,
      28574,
      624,
      21468,
      419,
      4565,
      3796,
      311,
      678,
      14158,
      315,
      279,
      6912,
      476,
      1101,
      3151,
      6174,
      438,
      10007,
      304,
      279,
      1467,
      3684,
      30,
      151645,
      198,
      151644,
      77091,
      198,
      1986,
      4565,
      1172,
      16790,
      311,
      11101,
      6816,
      323,
      50219,
      25326,
      14158,
      315,
      279,
      3772,
      5980,
      21386,
      10007,
      304,
      279,
      1467,
      3684,
      13,
      151645,
      198,
      151644,
      872,
      198,
      6713,
      498,
      8474,
      752,
      1526,
      279,
      1882,
      315,
      27362,
      279,
      14246,
      2168,
      19548,
      4565,
      389,
      847,
      11101,
      6816,
      323,
      50219,
      25326,
      14158,
      30,
      151645,
      198,
      151644,
      77091,
      198,
      39814,
      11,
      1588,
      525,
      279,
      7354,
      311,
      7283,
      279,
      14246,
      2168,
      19548,
      4565,
      389,
      697,
      11101,
      6816,
      323,
      50219,
      25326,
      14158,
      1447,
      16,
      13,
      2835,
      304,
      311,
      697,
      95025,
      2692,
      323,
      728,
      311,
      697,
      8105,
      9129,
      624,
      17,
      13,
      9189,
      389,
      46334,
      6912,
      369,
      279,
      3772,
      5980,
      6912,
      498,
      525,
      1667,
      624,
      18,
      13,
      81739,
      311,
      279,
      11101,
      6816,
      476,
      50219,
      25326,
      3772,
      1380,
      498,
      1366,
      311,
      7283,
      279,
      14246,
      2168,
      19548,
      4565,
      624,
      19,
      13,
      3085,
      279,
      3772,
      1787,
      11,
      4205,
      389,
      279,
      3772,
      594,
      6243,
      320,
      54727,
      8,
      4603,
      304,
      279,
      1909,
      7950,
      9131,
      624,
      20,
      13,
      758,
      279,
      5003,
      6945,
      429,
      7952,
      11,
      1401,
      369,
      458,
      2999,
      29829,
      364,
      1906,
      3037,
      6,
      476,
      364,
      1906,
      19548,
      23569,
      21,
      13,
      1416,
      2500,
      11,
      3293,
      364,
      7812,
      14246,
      2168,
      389,
      19548,
      23569,
      22,
      13,
      10255,
      279,
      4344,
      323,
      17124,
      279,
      11101,
      14,
      51943,
      11101,
      2150,
      311,
      1490,
      279,
      2456,
      382,
      2679,
      498,
      2299,
      3432,
      12264,
      9271,
      279,
      6243,
      11,
      279,
      1850,
      3166,
      311,
      653,
      374,
      8300,
      311,
      697,
      6912,
      594,
      9705,
      11,
      2474,
      279,
      3728,
      323,
      54302,
      315,
      5003,
      646,
      13289,
      1948,
      21386,
      13,
      151645,
      198,
      151644,
      872,
      198,
      6713,
      498,
      3410,
      752,
      448,
      264,
      2656,
      311,
      279,
      9705,
      369,
      847,
      6912,
      30,
      151645,
      198,
      151644,
      77091,
      198,
      40,
      1513,
      944,
      614,
      2615,
      311,
      697,
      3553,
      594,
      6912,
      1995,
      13,
      4354,
      11,
      498,
      646,
      5990,
      1477,
      279,
      9705,
      369,
      697,
      6912,
      553,
      2087,
      311,
      279,
      8061,
      1437,
      6912,
      3553,
      11,
      9271,
      697,
      6912,
      323,
      18458,
      389,
      279,
      364,
      23362,
      6,
      2656,
      7407,
      304,
      279,
      5622,
      1290,
      9131,
      315,
      279,
      2150,
      13,
      38478,
      11,
      498,
      646,
      653,
      264,
      11558,
      2711,
      369,
      279,
      829,
      315,
      697,
      6912,
      8110,
      553,
      364,
      76303,
      6,
      476,
      364,
      872,
      8474,
      4427,
      151645,
      198,
      151644,
      872,
      198,
      6713,
      498,
      7683,
      421,
      419,
      4565,
      1083,
      4278,
      369,
      279,
      17251,
      14028,
      3772,
      315,
      847,
      6912,
      30,
      151645,
      198,
      151644,
      77091,
      198,
      151667,
      271,
      151668,
      271,
      785,
      14246,
      2168,
      19548,
      4565,
      1231,
      476,
      1231,
      537,
      975,
      369,
      697,
      17251,
      14028,
      3772,
      11,
      11649,
      389,
      279,
      6546,
      315,
      697,
      6912,
      13,
      4329,
      21386,
      2924,
      419,
      4565,
      304,
      279,
      17251,
      14028,
      3772,
      553,
      1638,
      11,
      1393,
      3800,
      1231,
      1373,
      5107,
      48041,
      13,
      2014,
      1779,
      421,
      419,
      4565,
      374,
      2500,
      369,
      279,
      17251,
      14028,
      3772,
      315,
      697,
      6912,
      11,
      1795,
      1493,
      7354,
      1447,
      16,
      13,
      5994,
      311,
      279,
      17251,
      14028,
      3772,
      1380,
      498,
      1035,
      1075,
      311,
      7283,
      279,
      4565,
      13,
      220,
      17,
      13,
      9189,
      389,
      279,
      17251,
      14028,
      5003,
      4603,
      320,
      54727,
      4603,
      8,
      323,
      1401,
      369,
      364,
      1906,
      3037,
      6,
      476,
      364,
      1906,
      19548,
      4427,
      220,
      18,
      13,
      1416,
      2500,
      11,
      3293,
      364,
      7812,
      14246,
      2168,
      389,
      19548,
      4427,
      220,
      19,
      13,
      10255,
      279,
      4344,
      13,
      1416,
      419,
      2999,
      374,
      537,
      2500,
      304,
      697,
      17251,
      14028,
      3772,
      5003,
      11,
      498,
      1231,
      1184,
      311,
      5545,
      700,
      311,
      697,
      6912,
      15754,
      369,
      12994,
      448,
      2526,
      4849,
      697,
      17251,
      14028,
      3772,
      311,
      2924,
      419,
      4565,
      13,
      151645,
      198],
     'attention_mask': [1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1]}




```python

oneshot(
    model=model,         
    dataset=sampled_ds,
    recipe=recipe,
    max_seq_length=4096,
    num_calibration_samples=256,
    output_dir="./qwen-3-0-6b-comprssed"
)
```

    2026-10-07T07:06:19.1695 | __init__ | WARNING - Disabling tokenizer parallelism due to threading conflict between FastTokenizer and Datasets. Set TOKENIZERS_PARALLELISM=false to suppress this warning.
    2026-10-07T07:06:20.0205 | reset | INFO - Compression lifecycle reset
    2026-10-07T07:06:20.0435 | from_modifiers | INFO - Creating recipe from modifiers


    Applying quantization config: 100%|█████████████████████████████████████████| 196/196 [00:00<00:00, 590.56it/s]

    2026-10-07T07:06:20.4070 | initialize | INFO - Compression lifecycle initialized for 1 modifiers
    2026-10-07T07:06:20.4074 | IndependentPipeline | INFO - Inferred `SequentialPipeline` for `GPTQModifier`


    
    W1007 07:06:20.469000 76466 torch/fx/_symbolic_trace.py:57] is_fx_tracing will return true for both fx.symbolic_trace and torch.export. Please use is_fx_tracing_symbolic_tracing() for specifically fx.symbolic_trace or torch.compiler.is_compiling() for specifically torch.export/compile.
    Preparing cache: 100%|█████████████████████████████████████████████████████| 256/256 [00:00<00:00, 2559.14it/s]
    (1/29): Calibrating: 100%|███████████████████████████████████████████████████| 256/256 [00:03<00:00, 76.43it/s]
    (1/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [00:02<00:00, 103.72it/s]
    (2/29): Calibrating: 100%|███████████████████████████████████████████████████| 256/256 [00:40<00:00,  6.30it/s]


    2026-10-07T07:07:08.5060 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.0.self_attn.q_proj']
    2026-10-07T07:07:10.7209 | GPTQ | METRIC - time 2.20s
    2026-10-07T07:07:10.7215 | GPTQ | METRIC - error 475.96
    2026-10-07T07:07:10.7225 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:07:10.7291 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.0.self_attn.k_proj']
    2026-10-07T07:07:10.8951 | GPTQ | METRIC - time 0.17s
    2026-10-07T07:07:10.8955 | GPTQ | METRIC - error 209.49
    2026-10-07T07:07:10.8961 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:07:10.8982 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.0.self_attn.v_proj']
    2026-10-07T07:07:11.0092 | GPTQ | METRIC - time 0.11s
    2026-10-07T07:07:11.0096 | GPTQ | METRIC - error 159.59
    2026-10-07T07:07:11.0100 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:07:11.0158 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.0.self_attn.o_proj']
    2026-10-07T07:07:12.1780 | GPTQ | METRIC - time 1.16s
    2026-10-07T07:07:12.1785 | GPTQ | METRIC - error 94.94
    2026-10-07T07:07:12.1790 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:07:12.1851 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.0.mlp.gate_proj']
    2026-10-07T07:07:12.3326 | GPTQ | METRIC - time 0.15s
    2026-10-07T07:07:12.3330 | GPTQ | METRIC - error 3762.00
    2026-10-07T07:07:12.3334 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:07:12.3359 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.0.mlp.up_proj']
    2026-10-07T07:07:12.4741 | GPTQ | METRIC - time 0.14s
    2026-10-07T07:07:12.4744 | GPTQ | METRIC - error 1914.01
    2026-10-07T07:07:12.4748 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:07:12.4804 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.0.mlp.down_proj']
    2026-10-07T07:07:14.1182 | GPTQ | METRIC - time 1.64s
    2026-10-07T07:07:14.1217 | GPTQ | METRIC - error 119.77
    2026-10-07T07:07:14.1227 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (2/29): Propagating: 100%|███████████████████████████████████████████████████| 256/256 [00:22<00:00, 11.47it/s]
    (3/29): Calibrating: 100%|███████████████████████████████████████████████████| 256/256 [00:52<00:00,  4.89it/s]


    2026-10-07T07:08:29.8426 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.1.self_attn.q_proj']
    2026-10-07T07:08:30.0928 | GPTQ | METRIC - time 0.24s
    2026-10-07T07:08:30.0933 | GPTQ | METRIC - error 152.50
    2026-10-07T07:08:30.0970 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:08:30.1049 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.1.self_attn.k_proj']
    2026-10-07T07:08:30.2203 | GPTQ | METRIC - time 0.11s
    2026-10-07T07:08:30.2213 | GPTQ | METRIC - error 67.73
    2026-10-07T07:08:30.2218 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:08:30.2259 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.1.self_attn.v_proj']
    2026-10-07T07:08:30.3350 | GPTQ | METRIC - time 0.11s
    2026-10-07T07:08:30.3353 | GPTQ | METRIC - error 63.98
    2026-10-07T07:08:30.3357 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:08:30.3402 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.1.self_attn.o_proj']
    2026-10-07T07:08:30.6508 | GPTQ | METRIC - time 0.31s
    2026-10-07T07:08:30.6514 | GPTQ | METRIC - error 21.24
    2026-10-07T07:08:30.6529 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:08:30.6591 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.1.mlp.gate_proj']
    2026-10-07T07:08:30.8946 | GPTQ | METRIC - time 0.23s
    2026-10-07T07:08:30.8955 | GPTQ | METRIC - error 22621.72
    2026-10-07T07:08:30.8970 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:08:30.9127 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.1.mlp.up_proj']
    2026-10-07T07:08:31.1651 | GPTQ | METRIC - time 0.25s
    2026-10-07T07:08:31.1658 | GPTQ | METRIC - error 6672.28
    2026-10-07T07:08:31.1666 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:08:31.1732 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.1.mlp.down_proj']
    2026-10-07T07:08:31.8315 | GPTQ | METRIC - time 0.66s
    2026-10-07T07:08:31.8320 | GPTQ | METRIC - error 161.58
    2026-10-07T07:08:31.8324 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (3/29): Propagating: 100%|███████████████████████████████████████████████████| 256/256 [00:55<00:00,  4.65it/s]
    (4/29): Calibrating: 100%|███████████████████████████████████████████████████| 256/256 [01:43<00:00,  2.46it/s]


    2026-10-07T07:11:12.7005 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.2.self_attn.q_proj']
    2026-10-07T07:11:13.4405 | GPTQ | METRIC - time 0.70s
    2026-10-07T07:11:13.4478 | GPTQ | METRIC - error 299.54
    2026-10-07T07:11:13.4994 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:11:13.5587 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.2.self_attn.k_proj']
    2026-10-07T07:11:13.6935 | GPTQ | METRIC - time 0.13s
    2026-10-07T07:11:13.6940 | GPTQ | METRIC - error 127.19
    2026-10-07T07:11:13.6952 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:11:13.6991 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.2.self_attn.v_proj']
    2026-10-07T07:11:13.8236 | GPTQ | METRIC - time 0.12s
    2026-10-07T07:11:13.8241 | GPTQ | METRIC - error 123.14
    2026-10-07T07:11:13.8290 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:11:13.8888 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.2.self_attn.o_proj']
    2026-10-07T07:11:14.2850 | GPTQ | METRIC - time 0.40s
    2026-10-07T07:11:14.2855 | GPTQ | METRIC - error 34.81
    2026-10-07T07:11:14.2859 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:11:14.3070 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.2.mlp.gate_proj']
    2026-10-07T07:11:14.4583 | GPTQ | METRIC - time 0.15s
    2026-10-07T07:11:14.4641 | GPTQ | METRIC - error 15164.16
    2026-10-07T07:11:14.4646 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:11:14.4670 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.2.mlp.up_proj']
    2026-10-07T07:11:14.6170 | GPTQ | METRIC - time 0.15s
    2026-10-07T07:11:14.6175 | GPTQ | METRIC - error 5476.72
    2026-10-07T07:11:14.6191 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:11:14.6296 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.2.mlp.down_proj']
    2026-10-07T07:11:16.0378 | GPTQ | METRIC - time 1.41s
    2026-10-07T07:11:16.0389 | GPTQ | METRIC - error 4400.37
    2026-10-07T07:11:16.0397 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (4/29): Propagating: 100%|███████████████████████████████████████████████████| 256/256 [00:50<00:00,  5.02it/s]
    (5/29): Calibrating: 100%|███████████████████████████████████████████████████| 256/256 [02:03<00:00,  2.07it/s]


    2026-10-07T07:14:25.2951 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.3.self_attn.q_proj']
    2026-10-07T07:14:25.8989 | GPTQ | METRIC - time 0.45s
    2026-10-07T07:14:25.9011 | GPTQ | METRIC - error 2156.27
    2026-10-07T07:14:25.9181 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:14:25.9419 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.3.self_attn.k_proj']
    2026-10-07T07:14:26.1960 | GPTQ | METRIC - time 0.25s
    2026-10-07T07:14:26.2000 | GPTQ | METRIC - error 1059.72
    2026-10-07T07:14:26.2016 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:14:26.2204 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.3.self_attn.v_proj']
    2026-10-07T07:14:26.4159 | GPTQ | METRIC - time 0.17s
    2026-10-07T07:14:26.4181 | GPTQ | METRIC - error 1040.67
    2026-10-07T07:14:26.4193 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:14:26.4316 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.3.self_attn.o_proj']
    2026-10-07T07:14:26.7589 | GPTQ | METRIC - time 0.33s
    2026-10-07T07:14:26.7600 | GPTQ | METRIC - error 64.48
    2026-10-07T07:14:26.7619 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:14:26.7726 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.3.mlp.gate_proj']
    2026-10-07T07:14:27.0137 | GPTQ | METRIC - time 0.24s
    2026-10-07T07:14:27.0174 | GPTQ | METRIC - error 19688.86
    2026-10-07T07:14:27.0233 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:14:27.0298 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.3.mlp.up_proj']
    2026-10-07T07:14:27.3131 | GPTQ | METRIC - time 0.28s
    2026-10-07T07:14:27.3138 | GPTQ | METRIC - error 7609.83
    2026-10-07T07:14:27.3151 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:14:27.3307 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.3.mlp.down_proj']
    2026-10-07T07:14:30.7806 | GPTQ | METRIC - time 3.45s
    2026-10-07T07:14:30.7811 | GPTQ | METRIC - error 372.19
    2026-10-07T07:14:30.7843 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (5/29): Propagating: 100%|███████████████████████████████████████████████████| 256/256 [02:04<00:00,  2.06it/s]
    (6/29): Calibrating: 100%|███████████████████████████████████████████████████| 256/256 [00:52<00:00,  4.91it/s]


    2026-10-07T07:17:29.0735 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.4.self_attn.q_proj']
    2026-10-07T07:17:29.5706 | GPTQ | METRIC - time 0.40s
    2026-10-07T07:17:29.5738 | GPTQ | METRIC - error 1985.71
    2026-10-07T07:17:29.5804 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:17:29.5981 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.4.self_attn.k_proj']
    2026-10-07T07:17:29.7230 | GPTQ | METRIC - time 0.12s
    2026-10-07T07:17:29.7235 | GPTQ | METRIC - error 942.01
    2026-10-07T07:17:29.7247 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:17:29.7273 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.4.self_attn.v_proj']
    2026-10-07T07:17:29.8405 | GPTQ | METRIC - time 0.11s
    2026-10-07T07:17:29.8410 | GPTQ | METRIC - error 997.38
    2026-10-07T07:17:29.8419 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:17:29.8537 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.4.self_attn.o_proj']
    2026-10-07T07:17:30.1975 | GPTQ | METRIC - time 0.34s
    2026-10-07T07:17:30.1983 | GPTQ | METRIC - error 120.80
    2026-10-07T07:17:30.1991 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:17:30.2086 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.4.mlp.gate_proj']
    2026-10-07T07:17:30.4370 | GPTQ | METRIC - time 0.23s
    2026-10-07T07:17:30.4376 | GPTQ | METRIC - error 18083.41
    2026-10-07T07:17:30.4389 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:17:30.4429 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.4.mlp.up_proj']
    2026-10-07T07:17:30.7114 | GPTQ | METRIC - time 0.27s
    2026-10-07T07:17:30.7120 | GPTQ | METRIC - error 7872.97
    2026-10-07T07:17:30.7131 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:17:30.7237 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.4.mlp.down_proj']
    2026-10-07T07:17:31.4723 | GPTQ | METRIC - time 0.75s
    2026-10-07T07:17:31.4727 | GPTQ | METRIC - error 490.78
    2026-10-07T07:17:31.4732 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (6/29): Propagating: 100%|███████████████████████████████████████████████████| 256/256 [01:20<00:00,  3.17it/s]
    (7/29): Calibrating: 100%|███████████████████████████████████████████████████| 256/256 [03:30<00:00,  1.21it/s]


    2026-10-07T07:22:39.8569 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.5.self_attn.q_proj']
    2026-10-07T07:22:40.2341 | GPTQ | METRIC - time 0.35s
    2026-10-07T07:22:40.2350 | GPTQ | METRIC - error 3404.68
    2026-10-07T07:22:40.2415 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:22:40.2660 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.5.self_attn.k_proj']
    2026-10-07T07:22:40.7703 | GPTQ | METRIC - time 0.50s
    2026-10-07T07:22:40.7707 | GPTQ | METRIC - error 1399.71
    2026-10-07T07:22:40.7715 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:22:40.7749 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.5.self_attn.v_proj']
    2026-10-07T07:22:40.8847 | GPTQ | METRIC - time 0.11s
    2026-10-07T07:22:40.8855 | GPTQ | METRIC - error 1472.48
    2026-10-07T07:22:40.8869 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:22:40.8956 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.5.self_attn.o_proj']
    2026-10-07T07:22:41.1557 | GPTQ | METRIC - time 0.26s
    2026-10-07T07:22:41.1562 | GPTQ | METRIC - error 159.77
    2026-10-07T07:22:41.1566 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:22:41.1970 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.5.mlp.gate_proj']
    2026-10-07T07:22:41.3459 | GPTQ | METRIC - time 0.15s
    2026-10-07T07:22:41.3486 | GPTQ | METRIC - error 12464.25
    2026-10-07T07:22:41.3491 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:22:41.3538 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.5.mlp.up_proj']
    2026-10-07T07:22:41.4922 | GPTQ | METRIC - time 0.14s
    2026-10-07T07:22:41.4975 | GPTQ | METRIC - error 7372.17
    2026-10-07T07:22:41.4979 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:22:41.5381 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.5.mlp.down_proj']
    2026-10-07T07:22:43.8011 | GPTQ | METRIC - time 2.26s
    2026-10-07T07:22:43.8031 | GPTQ | METRIC - error 540.49
    2026-10-07T07:22:43.8040 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (7/29): Propagating: 100%|███████████████████████████████████████████████████| 256/256 [05:17<00:00,  1.24s/it]
    (8/29): Calibrating: 100%|███████████████████████████████████████████████████| 256/256 [07:57<00:00,  1.87s/it]


    2026-10-07T07:36:28.0128 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.6.self_attn.q_proj']
    2026-10-07T07:36:28.5391 | GPTQ | METRIC - time 0.44s
    2026-10-07T07:36:28.5404 | GPTQ | METRIC - error 2347.91
    2026-10-07T07:36:28.5450 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:36:28.5582 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.6.self_attn.k_proj']
    2026-10-07T07:36:29.4716 | GPTQ | METRIC - time 0.91s
    2026-10-07T07:36:29.4721 | GPTQ | METRIC - error 1056.59
    2026-10-07T07:36:29.4727 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:36:29.4759 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.6.self_attn.v_proj']
    2026-10-07T07:36:29.5933 | GPTQ | METRIC - time 0.12s
    2026-10-07T07:36:29.5939 | GPTQ | METRIC - error 980.61
    2026-10-07T07:36:29.5955 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:36:29.6074 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.6.self_attn.o_proj']
    2026-10-07T07:36:29.9393 | GPTQ | METRIC - time 0.33s
    2026-10-07T07:36:29.9397 | GPTQ | METRIC - error 144.44
    2026-10-07T07:36:29.9402 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:36:29.9512 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.6.mlp.gate_proj']
    2026-10-07T07:36:30.5830 | GPTQ | METRIC - time 0.63s
    2026-10-07T07:36:30.5835 | GPTQ | METRIC - error 14319.30
    2026-10-07T07:36:30.5845 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:36:30.5874 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.6.mlp.up_proj']
    2026-10-07T07:36:30.7350 | GPTQ | METRIC - time 0.15s
    2026-10-07T07:36:30.7355 | GPTQ | METRIC - error 8892.73
    2026-10-07T07:36:30.7377 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:36:30.7597 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.6.mlp.down_proj']
    2026-10-07T07:36:36.7466 | GPTQ | METRIC - time 5.99s
    2026-10-07T07:36:36.7511 | GPTQ | METRIC - error 664.73
    2026-10-07T07:36:36.7538 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (8/29): Propagating: 100%|███████████████████████████████████████████████████| 256/256 [09:49<00:00,  2.30s/it]
    (9/29): Calibrating: 100%|███████████████████████████████████████████████████| 256/256 [08:57<00:00,  2.10s/it]


    2026-10-07T07:55:34.7532 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.7.self_attn.q_proj']
    2026-10-07T07:55:35.5859 | GPTQ | METRIC - time 0.56s
    2026-10-07T07:55:35.5865 | GPTQ | METRIC - error 4431.20
    2026-10-07T07:55:35.5945 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:55:35.6078 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.7.self_attn.k_proj']
    2026-10-07T07:55:36.0652 | GPTQ | METRIC - time 0.46s
    2026-10-07T07:55:36.0656 | GPTQ | METRIC - error 1841.03
    2026-10-07T07:55:36.0668 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:55:36.0698 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.7.self_attn.v_proj']
    2026-10-07T07:55:36.1961 | GPTQ | METRIC - time 0.13s
    2026-10-07T07:55:36.1966 | GPTQ | METRIC - error 2004.21
    2026-10-07T07:55:36.1982 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:55:36.2080 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.7.self_attn.o_proj']
    2026-10-07T07:55:37.1445 | GPTQ | METRIC - time 0.94s
    2026-10-07T07:55:37.1472 | GPTQ | METRIC - error 225.80
    2026-10-07T07:55:37.2160 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:55:37.3206 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.7.mlp.gate_proj']
    2026-10-07T07:55:37.5989 | GPTQ | METRIC - time 0.28s
    2026-10-07T07:55:37.5993 | GPTQ | METRIC - error 15774.34
    2026-10-07T07:55:37.6020 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:55:37.6091 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.7.mlp.up_proj']
    2026-10-07T07:55:37.8211 | GPTQ | METRIC - time 0.21s
    2026-10-07T07:55:37.8218 | GPTQ | METRIC - error 9641.10
    2026-10-07T07:55:37.8225 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T07:55:37.8984 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.7.mlp.down_proj']
    2026-10-07T07:55:39.5367 | GPTQ | METRIC - time 1.64s
    2026-10-07T07:55:39.5529 | GPTQ | METRIC - error 763.42
    2026-10-07T07:55:39.5573 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (9/29): Propagating: 100%|███████████████████████████████████████████████████| 256/256 [10:47<00:00,  2.53s/it]
    (10/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [08:45<00:00,  2.05s/it]


    2026-10-07T08:15:35.7832 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.8.self_attn.q_proj']
    2026-10-07T08:15:36.6429 | GPTQ | METRIC - time 0.80s
    2026-10-07T08:15:36.6439 | GPTQ | METRIC - error 5218.90
    2026-10-07T08:15:36.6543 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:15:36.6709 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.8.self_attn.k_proj']
    2026-10-07T08:15:37.5720 | GPTQ | METRIC - time 0.90s
    2026-10-07T08:15:37.5724 | GPTQ | METRIC - error 2377.18
    2026-10-07T08:15:37.5728 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:15:37.5755 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.8.self_attn.v_proj']
    2026-10-07T08:15:37.6906 | GPTQ | METRIC - time 0.11s
    2026-10-07T08:15:37.6916 | GPTQ | METRIC - error 2158.51
    2026-10-07T08:15:37.6928 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:15:37.7077 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.8.self_attn.o_proj']
    2026-10-07T08:15:40.0327 | GPTQ | METRIC - time 2.32s
    2026-10-07T08:15:40.0335 | GPTQ | METRIC - error 236.35
    2026-10-07T08:15:40.0347 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:15:40.0428 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.8.mlp.gate_proj']
    2026-10-07T08:15:41.3386 | GPTQ | METRIC - time 1.30s
    2026-10-07T08:15:41.3400 | GPTQ | METRIC - error 15066.73
    2026-10-07T08:15:41.3405 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:15:41.3448 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.8.mlp.up_proj']
    2026-10-07T08:15:41.5767 | GPTQ | METRIC - time 0.23s
    2026-10-07T08:15:41.5773 | GPTQ | METRIC - error 9582.04
    2026-10-07T08:15:41.5780 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:15:41.5888 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.8.mlp.down_proj']
    2026-10-07T08:15:44.8370 | GPTQ | METRIC - time 3.25s
    2026-10-07T08:15:44.8383 | GPTQ | METRIC - error 751.48
    2026-10-07T08:15:44.8391 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (10/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [12:22<00:00,  2.90s/it]
    (11/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [11:10<00:00,  2.62s/it]


    2026-10-07T08:39:49.2194 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.9.self_attn.q_proj']
    2026-10-07T08:39:49.9985 | GPTQ | METRIC - time 0.70s
    2026-10-07T08:39:50.0001 | GPTQ | METRIC - error 9211.09
    2026-10-07T08:39:50.0143 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:39:50.0398 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.9.self_attn.k_proj']
    2026-10-07T08:39:50.4328 | GPTQ | METRIC - time 0.39s
    2026-10-07T08:39:50.4333 | GPTQ | METRIC - error 3826.44
    2026-10-07T08:39:50.4343 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:39:50.4373 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.9.self_attn.v_proj']
    2026-10-07T08:39:50.5486 | GPTQ | METRIC - time 0.11s
    2026-10-07T08:39:50.5490 | GPTQ | METRIC - error 3775.65
    2026-10-07T08:39:50.5501 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:39:50.5622 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.9.self_attn.o_proj']
    2026-10-07T08:39:51.2140 | GPTQ | METRIC - time 0.65s
    2026-10-07T08:39:51.2144 | GPTQ | METRIC - error 413.76
    2026-10-07T08:39:51.2155 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:39:51.2259 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.9.mlp.gate_proj']
    2026-10-07T08:39:51.3728 | GPTQ | METRIC - time 0.15s
    2026-10-07T08:39:51.3732 | GPTQ | METRIC - error 15519.60
    2026-10-07T08:39:51.3736 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:39:51.3761 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.9.mlp.up_proj']
    2026-10-07T08:39:51.7305 | GPTQ | METRIC - time 0.35s
    2026-10-07T08:39:51.7309 | GPTQ | METRIC - error 9606.82
    2026-10-07T08:39:51.7335 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T08:39:51.7588 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.9.mlp.down_proj']
    2026-10-07T08:39:57.4780 | GPTQ | METRIC - time 5.72s
    2026-10-07T08:39:57.4814 | GPTQ | METRIC - error 817.75
    2026-10-07T08:39:57.4835 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (11/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [09:41<00:00,  2.27s/it]
    (12/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [11:23<00:00,  2.67s/it]


    2026-10-07T09:01:24.0262 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.10.self_attn.q_proj']
    2026-10-07T09:01:24.5859 | GPTQ | METRIC - time 0.50s
    2026-10-07T09:01:24.5867 | GPTQ | METRIC - error 6914.89
    2026-10-07T09:01:24.5943 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:01:24.6071 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.10.self_attn.k_proj']
    2026-10-07T09:01:24.9689 | GPTQ | METRIC - time 0.36s
    2026-10-07T09:01:24.9693 | GPTQ | METRIC - error 2912.81
    2026-10-07T09:01:24.9698 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:01:24.9724 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.10.self_attn.v_proj']
    2026-10-07T09:01:25.0886 | GPTQ | METRIC - time 0.12s
    2026-10-07T09:01:25.0891 | GPTQ | METRIC - error 2909.41
    2026-10-07T09:01:25.0922 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:01:25.1035 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.10.self_attn.o_proj']
    2026-10-07T09:01:25.6810 | GPTQ | METRIC - time 0.58s
    2026-10-07T09:01:25.6833 | GPTQ | METRIC - error 341.49
    2026-10-07T09:01:25.6838 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:01:25.6908 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.10.mlp.gate_proj']
    2026-10-07T09:01:25.8428 | GPTQ | METRIC - time 0.15s
    2026-10-07T09:01:25.8432 | GPTQ | METRIC - error 14921.25
    2026-10-07T09:01:25.8436 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:01:25.8462 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.10.mlp.up_proj']
    2026-10-07T09:01:26.2529 | GPTQ | METRIC - time 0.41s
    2026-10-07T09:01:26.2533 | GPTQ | METRIC - error 9292.13
    2026-10-07T09:01:26.2594 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:01:26.2692 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.10.mlp.down_proj']
    2026-10-07T09:01:29.8588 | GPTQ | METRIC - time 3.59s
    2026-10-07T09:01:29.8602 | GPTQ | METRIC - error 1097.43
    2026-10-07T09:01:29.8619 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (12/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [09:45<00:00,  2.29s/it]
    (13/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [11:57<00:00,  2.80s/it]


    2026-10-07T09:23:36.1126 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.11.self_attn.q_proj']
    2026-10-07T09:23:36.6674 | GPTQ | METRIC - time 0.51s
    2026-10-07T09:23:36.6688 | GPTQ | METRIC - error 14563.98
    2026-10-07T09:23:36.6764 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:23:36.6880 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.11.self_attn.k_proj']
    2026-10-07T09:23:37.0336 | GPTQ | METRIC - time 0.34s
    2026-10-07T09:23:37.0342 | GPTQ | METRIC - error 5760.35
    2026-10-07T09:23:37.0354 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:23:37.0381 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.11.self_attn.v_proj']
    2026-10-07T09:23:37.1504 | GPTQ | METRIC - time 0.11s
    2026-10-07T09:23:37.1509 | GPTQ | METRIC - error 5066.56
    2026-10-07T09:23:37.1519 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:23:37.1608 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.11.self_attn.o_proj']
    2026-10-07T09:23:37.6013 | GPTQ | METRIC - time 0.44s
    2026-10-07T09:23:37.6017 | GPTQ | METRIC - error 788.70
    2026-10-07T09:23:37.6021 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:23:37.6092 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.11.mlp.gate_proj']
    2026-10-07T09:23:37.7554 | GPTQ | METRIC - time 0.15s
    2026-10-07T09:23:37.7558 | GPTQ | METRIC - error 10421.28
    2026-10-07T09:23:37.7568 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:23:37.7605 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.11.mlp.up_proj']
    2026-10-07T09:23:38.1007 | GPTQ | METRIC - time 0.34s
    2026-10-07T09:23:38.1015 | GPTQ | METRIC - error 7875.65
    2026-10-07T09:23:38.1020 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T09:23:38.1120 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.11.mlp.down_proj']
    2026-10-07T09:23:39.8566 | GPTQ | METRIC - time 1.74s
    2026-10-07T09:23:39.8577 | GPTQ | METRIC - error 1085.14
    2026-10-07T09:23:39.8585 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (13/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [10:27<00:00,  2.45s/it]
    (14/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [32:05<00:00,  7.52s/it]


    2026-10-07T10:06:26.0278 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.12.self_attn.q_proj']
    2026-10-07T10:06:26.5858 | GPTQ | METRIC - time 0.47s
    2026-10-07T10:06:26.5864 | GPTQ | METRIC - error 11271.79
    2026-10-07T10:06:26.5944 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:06:26.6085 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.12.self_attn.k_proj']
    2026-10-07T10:06:27.0205 | GPTQ | METRIC - time 0.41s
    2026-10-07T10:06:27.0210 | GPTQ | METRIC - error 4204.93
    2026-10-07T10:06:27.0221 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:06:27.0258 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.12.self_attn.v_proj']
    2026-10-07T10:06:27.1369 | GPTQ | METRIC - time 0.11s
    2026-10-07T10:06:27.1377 | GPTQ | METRIC - error 4185.51
    2026-10-07T10:06:27.1394 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:06:27.1502 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.12.self_attn.o_proj']
    2026-10-07T10:06:27.9956 | GPTQ | METRIC - time 0.84s
    2026-10-07T10:06:27.9962 | GPTQ | METRIC - error 417.45
    2026-10-07T10:06:27.9975 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:06:28.0067 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.12.mlp.gate_proj']
    2026-10-07T10:06:28.1876 | GPTQ | METRIC - time 0.18s
    2026-10-07T10:06:28.1880 | GPTQ | METRIC - error 9403.71
    2026-10-07T10:06:28.1891 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:06:28.1931 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.12.mlp.up_proj']
    2026-10-07T10:06:28.4086 | GPTQ | METRIC - time 0.21s
    2026-10-07T10:06:28.4091 | GPTQ | METRIC - error 7669.00
    2026-10-07T10:06:28.4097 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:06:28.4215 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.12.mlp.down_proj']
    2026-10-07T10:06:32.3424 | GPTQ | METRIC - time 3.92s
    2026-10-07T10:06:32.3439 | GPTQ | METRIC - error 1097.44
    2026-10-07T10:06:32.3452 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (14/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [10:34<00:00,  2.48s/it]
    (15/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [10:09<00:00,  2.38s/it]


    2026-10-07T10:27:43.8508 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.13.self_attn.q_proj']
    2026-10-07T10:27:44.9599 | GPTQ | METRIC - time 1.04s
    2026-10-07T10:27:44.9609 | GPTQ | METRIC - error 12372.42
    2026-10-07T10:27:44.9675 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:27:44.9844 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.13.self_attn.k_proj']
    2026-10-07T10:27:45.4914 | GPTQ | METRIC - time 0.51s
    2026-10-07T10:27:45.4919 | GPTQ | METRIC - error 4450.49
    2026-10-07T10:27:45.4924 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:27:45.5010 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.13.self_attn.v_proj']
    2026-10-07T10:27:45.6643 | GPTQ | METRIC - time 0.14s
    2026-10-07T10:27:45.6650 | GPTQ | METRIC - error 4913.03
    2026-10-07T10:27:45.6657 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:27:45.7277 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.13.self_attn.o_proj']
    2026-10-07T10:27:46.8001 | GPTQ | METRIC - time 1.07s
    2026-10-07T10:27:46.8009 | GPTQ | METRIC - error 575.09
    2026-10-07T10:27:46.8014 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:27:46.8105 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.13.mlp.gate_proj']
    2026-10-07T10:27:47.0141 | GPTQ | METRIC - time 0.20s
    2026-10-07T10:27:47.0147 | GPTQ | METRIC - error 10792.46
    2026-10-07T10:27:47.0154 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:27:47.0187 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.13.mlp.up_proj']
    2026-10-07T10:27:47.3501 | GPTQ | METRIC - time 0.33s
    2026-10-07T10:27:47.3523 | GPTQ | METRIC - error 8639.25
    2026-10-07T10:27:47.3569 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:27:47.3764 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.13.mlp.down_proj']
    2026-10-07T10:27:51.7827 | GPTQ | METRIC - time 4.40s
    2026-10-07T10:27:51.7955 | GPTQ | METRIC - error 1325.59
    2026-10-07T10:27:51.8018 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (15/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [10:55<00:00,  2.56s/it]
    (16/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [10:49<00:00,  2.54s/it]


    2026-10-07T10:49:58.4591 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.14.self_attn.q_proj']
    2026-10-07T10:49:59.0541 | GPTQ | METRIC - time 0.53s
    2026-10-07T10:49:59.0549 | GPTQ | METRIC - error 16795.76
    2026-10-07T10:49:59.0619 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:49:59.0788 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.14.self_attn.k_proj']
    2026-10-07T10:49:59.7380 | GPTQ | METRIC - time 0.66s
    2026-10-07T10:49:59.7385 | GPTQ | METRIC - error 6318.85
    2026-10-07T10:49:59.7394 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:49:59.7428 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.14.self_attn.v_proj']
    2026-10-07T10:49:59.8545 | GPTQ | METRIC - time 0.11s
    2026-10-07T10:49:59.8549 | GPTQ | METRIC - error 6179.94
    2026-10-07T10:49:59.8554 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:49:59.8675 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.14.self_attn.o_proj']
    2026-10-07T10:50:00.4514 | GPTQ | METRIC - time 0.58s
    2026-10-07T10:50:00.4527 | GPTQ | METRIC - error 639.31
    2026-10-07T10:50:00.4540 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:50:00.4649 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.14.mlp.gate_proj']
    2026-10-07T10:50:00.6127 | GPTQ | METRIC - time 0.15s
    2026-10-07T10:50:00.6131 | GPTQ | METRIC - error 11188.21
    2026-10-07T10:50:00.6140 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:50:00.6170 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.14.mlp.up_proj']
    2026-10-07T10:50:00.7544 | GPTQ | METRIC - time 0.14s
    2026-10-07T10:50:00.7547 | GPTQ | METRIC - error 9095.09
    2026-10-07T10:50:00.7551 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T10:50:00.7645 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.14.mlp.down_proj']
    2026-10-07T10:50:05.4146 | GPTQ | METRIC - time 4.65s
    2026-10-07T10:50:05.4176 | GPTQ | METRIC - error 1579.52
    2026-10-07T10:50:05.4186 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (16/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [08:47<00:00,  2.06s/it]
    (17/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [12:00<00:00,  2.81s/it]


    2026-10-07T11:11:10.1718 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.15.self_attn.q_proj']
    2026-10-07T11:11:10.7937 | GPTQ | METRIC - time 0.58s
    2026-10-07T11:11:10.7944 | GPTQ | METRIC - error 31547.45
    2026-10-07T11:11:10.8053 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:11:10.8323 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.15.self_attn.k_proj']
    2026-10-07T11:11:11.3935 | GPTQ | METRIC - time 0.56s
    2026-10-07T11:11:11.3939 | GPTQ | METRIC - error 10169.01
    2026-10-07T11:11:11.3948 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:11:11.4128 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.15.self_attn.v_proj']
    2026-10-07T11:11:11.5287 | GPTQ | METRIC - time 0.11s
    2026-10-07T11:11:11.5291 | GPTQ | METRIC - error 12568.01
    2026-10-07T11:11:11.5300 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:11:11.5407 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.15.self_attn.o_proj']
    2026-10-07T11:11:12.0249 | GPTQ | METRIC - time 0.48s
    2026-10-07T11:11:12.0258 | GPTQ | METRIC - error 863.12
    2026-10-07T11:11:12.0262 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:11:12.0356 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.15.mlp.gate_proj']
    2026-10-07T11:11:12.2173 | GPTQ | METRIC - time 0.18s
    2026-10-07T11:11:12.2180 | GPTQ | METRIC - error 13226.72
    2026-10-07T11:11:12.2214 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:11:12.2285 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.15.mlp.up_proj']
    2026-10-07T11:11:12.5503 | GPTQ | METRIC - time 0.32s
    2026-10-07T11:11:12.5516 | GPTQ | METRIC - error 10910.34
    2026-10-07T11:11:12.5535 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:11:12.5711 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.15.mlp.down_proj']
    2026-10-07T11:11:17.2874 | GPTQ | METRIC - time 4.71s
    2026-10-07T11:11:17.2911 | GPTQ | METRIC - error 2240.13
    2026-10-07T11:11:17.2940 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (17/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [11:11<00:00,  2.62s/it]
    (18/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [09:15<00:00,  2.17s/it]


    2026-10-07T11:31:52.4306 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.16.self_attn.q_proj']
    2026-10-07T11:31:53.0444 | GPTQ | METRIC - time 0.53s
    2026-10-07T11:31:53.0456 | GPTQ | METRIC - error 44347.54
    2026-10-07T11:31:53.0569 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:31:53.0930 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.16.self_attn.k_proj']
    2026-10-07T11:31:53.8636 | GPTQ | METRIC - time 0.77s
    2026-10-07T11:31:53.8641 | GPTQ | METRIC - error 15763.18
    2026-10-07T11:31:53.8649 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:31:53.8687 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.16.self_attn.v_proj']
    2026-10-07T11:31:53.9783 | GPTQ | METRIC - time 0.11s
    2026-10-07T11:31:53.9787 | GPTQ | METRIC - error 14426.15
    2026-10-07T11:31:53.9796 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:31:53.9872 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.16.self_attn.o_proj']
    2026-10-07T11:31:54.6153 | GPTQ | METRIC - time 0.63s
    2026-10-07T11:31:54.6158 | GPTQ | METRIC - error 1203.17
    2026-10-07T11:31:54.6167 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:31:54.6271 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.16.mlp.gate_proj']
    2026-10-07T11:31:54.7698 | GPTQ | METRIC - time 0.14s
    2026-10-07T11:31:54.7703 | GPTQ | METRIC - error 14645.34
    2026-10-07T11:31:54.7711 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:31:54.7737 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.16.mlp.up_proj']
    2026-10-07T11:31:54.9123 | GPTQ | METRIC - time 0.14s
    2026-10-07T11:31:54.9127 | GPTQ | METRIC - error 13212.20
    2026-10-07T11:31:54.9134 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:31:54.9223 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.16.mlp.down_proj']
    2026-10-07T11:31:59.0540 | GPTQ | METRIC - time 4.13s
    2026-10-07T11:31:59.0558 | GPTQ | METRIC - error 5719.73
    2026-10-07T11:31:59.0583 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (18/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [11:10<00:00,  2.62s/it]
    (19/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [12:58<00:00,  3.04s/it]


    2026-10-07T11:56:28.0418 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.17.self_attn.q_proj']
    2026-10-07T11:56:28.8447 | GPTQ | METRIC - time 0.71s
    2026-10-07T11:56:28.8455 | GPTQ | METRIC - error 108554.35
    2026-10-07T11:56:28.8709 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:56:28.9166 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.17.self_attn.k_proj']
    2026-10-07T11:56:29.0896 | GPTQ | METRIC - time 0.17s
    2026-10-07T11:56:29.0905 | GPTQ | METRIC - error 36251.31
    2026-10-07T11:56:29.0933 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:56:29.1029 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.17.self_attn.v_proj']
    2026-10-07T11:56:29.4134 | GPTQ | METRIC - time 0.31s
    2026-10-07T11:56:29.4139 | GPTQ | METRIC - error 42501.01
    2026-10-07T11:56:29.4153 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:56:29.4643 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.17.self_attn.o_proj']
    2026-10-07T11:56:30.5628 | GPTQ | METRIC - time 1.10s
    2026-10-07T11:56:30.5634 | GPTQ | METRIC - error 2898.92
    2026-10-07T11:56:30.5652 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:56:30.5744 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.17.mlp.gate_proj']
    2026-10-07T11:56:31.1062 | GPTQ | METRIC - time 0.53s
    2026-10-07T11:56:31.1102 | GPTQ | METRIC - error 22650.18
    2026-10-07T11:56:31.1110 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:56:31.1157 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.17.mlp.up_proj']
    2026-10-07T11:56:31.6542 | GPTQ | METRIC - time 0.54s
    2026-10-07T11:56:31.6563 | GPTQ | METRIC - error 20522.90
    2026-10-07T11:56:31.6580 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T11:56:31.6798 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.17.mlp.down_proj']
    2026-10-07T11:56:37.2499 | GPTQ | METRIC - time 5.57s
    2026-10-07T11:56:37.2593 | GPTQ | METRIC - error 7969.86
    2026-10-07T11:56:37.2615 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (19/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [10:58<00:00,  2.57s/it]
    (20/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [09:37<00:00,  2.25s/it]


    2026-10-07T12:17:18.8384 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.18.self_attn.q_proj']
    2026-10-07T12:17:19.1564 | GPTQ | METRIC - time 0.29s
    2026-10-07T12:17:19.1576 | GPTQ | METRIC - error 112394.97
    2026-10-07T12:17:19.1621 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:17:19.1769 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.18.self_attn.k_proj']
    2026-10-07T12:17:19.2952 | GPTQ | METRIC - time 0.12s
    2026-10-07T12:17:19.2956 | GPTQ | METRIC - error 36038.30
    2026-10-07T12:17:19.2966 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:17:19.2987 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.18.self_attn.v_proj']
    2026-10-07T12:17:19.4103 | GPTQ | METRIC - time 0.11s
    2026-10-07T12:17:19.4108 | GPTQ | METRIC - error 42600.62
    2026-10-07T12:17:19.4112 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:17:19.4219 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.18.self_attn.o_proj']
    2026-10-07T12:17:19.9146 | GPTQ | METRIC - time 0.49s
    2026-10-07T12:17:19.9151 | GPTQ | METRIC - error 2599.73
    2026-10-07T12:17:19.9206 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:17:19.9661 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.18.mlp.gate_proj']
    2026-10-07T12:17:20.1195 | GPTQ | METRIC - time 0.15s
    2026-10-07T12:17:20.1199 | GPTQ | METRIC - error 28494.56
    2026-10-07T12:17:20.1204 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:17:20.1228 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.18.mlp.up_proj']
    2026-10-07T12:17:20.3040 | GPTQ | METRIC - time 0.18s
    2026-10-07T12:17:20.3048 | GPTQ | METRIC - error 26314.64
    2026-10-07T12:17:20.3055 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:17:20.3141 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.18.mlp.down_proj']
    2026-10-07T12:17:23.7152 | GPTQ | METRIC - time 3.40s
    2026-10-07T12:17:23.7157 | GPTQ | METRIC - error 12674.88
    2026-10-07T12:17:23.7161 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (20/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [08:31<00:00,  2.00s/it]
    (21/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [08:50<00:00,  2.07s/it]


    2026-10-07T12:34:54.3300 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.19.self_attn.q_proj']
    2026-10-07T12:34:54.8126 | GPTQ | METRIC - time 0.42s
    2026-10-07T12:34:54.8132 | GPTQ | METRIC - error 210287.91
    2026-10-07T12:34:54.8170 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:34:54.8278 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.19.self_attn.k_proj']
    2026-10-07T12:34:55.2478 | GPTQ | METRIC - time 0.42s
    2026-10-07T12:34:55.2486 | GPTQ | METRIC - error 63931.04
    2026-10-07T12:34:55.2491 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:34:55.2525 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.19.self_attn.v_proj']
    2026-10-07T12:34:55.3618 | GPTQ | METRIC - time 0.11s
    2026-10-07T12:34:55.3622 | GPTQ | METRIC - error 78923.94
    2026-10-07T12:34:55.3686 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:34:55.3786 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.19.self_attn.o_proj']
    2026-10-07T12:34:55.6665 | GPTQ | METRIC - time 0.29s
    2026-10-07T12:34:55.6670 | GPTQ | METRIC - error 4930.05
    2026-10-07T12:34:55.6679 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:34:55.6745 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.19.mlp.gate_proj']
    2026-10-07T12:34:55.8177 | GPTQ | METRIC - time 0.14s
    2026-10-07T12:34:55.8181 | GPTQ | METRIC - error 32568.75
    2026-10-07T12:34:55.8185 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:34:55.8209 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.19.mlp.up_proj']
    2026-10-07T12:34:55.9601 | GPTQ | METRIC - time 0.14s
    2026-10-07T12:34:55.9603 | GPTQ | METRIC - error 35056.98
    2026-10-07T12:34:55.9608 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:34:55.9689 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.19.mlp.down_proj']
    2026-10-07T12:34:59.7022 | GPTQ | METRIC - time 3.73s
    2026-10-07T12:34:59.7029 | GPTQ | METRIC - error 25363.69
    2026-10-07T12:34:59.7042 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (21/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [07:51<00:00,  1.84s/it]
    (22/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [07:02<00:00,  1.65s/it]


    2026-10-07T12:50:03.4292 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.20.self_attn.q_proj']
    2026-10-07T12:50:03.7834 | GPTQ | METRIC - time 0.33s
    2026-10-07T12:50:03.7839 | GPTQ | METRIC - error 294301.69
    2026-10-07T12:50:03.7905 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:50:03.8021 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.20.self_attn.k_proj']
    2026-10-07T12:50:03.9267 | GPTQ | METRIC - time 0.12s
    2026-10-07T12:50:03.9280 | GPTQ | METRIC - error 98633.81
    2026-10-07T12:50:03.9285 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:50:03.9311 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.20.self_attn.v_proj']
    2026-10-07T12:50:04.0422 | GPTQ | METRIC - time 0.11s
    2026-10-07T12:50:04.0427 | GPTQ | METRIC - error 119023.06
    2026-10-07T12:50:04.0437 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:50:04.0525 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.20.self_attn.o_proj']
    2026-10-07T12:50:04.3086 | GPTQ | METRIC - time 0.26s
    2026-10-07T12:50:04.3091 | GPTQ | METRIC - error 5429.12
    2026-10-07T12:50:04.3105 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:50:04.3199 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.20.mlp.gate_proj']
    2026-10-07T12:50:04.4625 | GPTQ | METRIC - time 0.14s
    2026-10-07T12:50:04.4628 | GPTQ | METRIC - error 38510.29
    2026-10-07T12:50:04.4632 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:50:04.4656 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.20.mlp.up_proj']
    2026-10-07T12:50:04.6054 | GPTQ | METRIC - time 0.14s
    2026-10-07T12:50:04.6058 | GPTQ | METRIC - error 43514.93
    2026-10-07T12:50:04.6062 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T12:50:04.6580 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.20.mlp.down_proj']
    2026-10-07T12:50:09.9302 | GPTQ | METRIC - time 5.27s
    2026-10-07T12:50:09.9306 | GPTQ | METRIC - error 32491.52
    2026-10-07T12:50:09.9366 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (22/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [07:13<00:00,  1.69s/it]
    (23/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [09:10<00:00,  2.15s/it]


    2026-10-07T13:06:49.0194 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.21.self_attn.q_proj']
    2026-10-07T13:06:49.4843 | GPTQ | METRIC - time 0.41s
    2026-10-07T13:06:49.4850 | GPTQ | METRIC - error 536165.75
    2026-10-07T13:06:49.4918 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:06:49.5039 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.21.self_attn.k_proj']
    2026-10-07T13:06:49.6815 | GPTQ | METRIC - time 0.18s
    2026-10-07T13:06:49.6832 | GPTQ | METRIC - error 176204.19
    2026-10-07T13:06:49.6837 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:06:49.6917 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.21.self_attn.v_proj']
    2026-10-07T13:06:49.8071 | GPTQ | METRIC - time 0.11s
    2026-10-07T13:06:49.8075 | GPTQ | METRIC - error 216480.88
    2026-10-07T13:06:49.8079 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:06:49.8186 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.21.self_attn.o_proj']
    2026-10-07T13:06:50.5440 | GPTQ | METRIC - time 0.72s
    2026-10-07T13:06:50.5445 | GPTQ | METRIC - error 7632.35
    2026-10-07T13:06:50.5464 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:06:50.5538 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.21.mlp.gate_proj']
    2026-10-07T13:06:50.6994 | GPTQ | METRIC - time 0.15s
    2026-10-07T13:06:50.6998 | GPTQ | METRIC - error 44005.82
    2026-10-07T13:06:50.7001 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:06:50.7039 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.21.mlp.up_proj']
    2026-10-07T13:06:50.8434 | GPTQ | METRIC - time 0.14s
    2026-10-07T13:06:50.8438 | GPTQ | METRIC - error 55520.64
    2026-10-07T13:06:50.8453 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:06:50.8542 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.21.mlp.down_proj']
    2026-10-07T13:06:51.9168 | GPTQ | METRIC - time 1.06s
    2026-10-07T13:06:51.9174 | GPTQ | METRIC - error 48631.23
    2026-10-07T13:06:51.9186 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (23/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [10:21<00:00,  2.43s/it]
    (24/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [09:45<00:00,  2.29s/it]


    2026-10-07T13:27:11.6610 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.22.self_attn.q_proj']
    2026-10-07T13:27:12.2625 | GPTQ | METRIC - time 0.51s
    2026-10-07T13:27:12.2641 | GPTQ | METRIC - error 594145.75
    2026-10-07T13:27:12.3026 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:27:12.3291 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.22.self_attn.k_proj']
    2026-10-07T13:27:13.0003 | GPTQ | METRIC - time 0.67s
    2026-10-07T13:27:13.0027 | GPTQ | METRIC - error 209944.44
    2026-10-07T13:27:13.0038 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:27:13.0076 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.22.self_attn.v_proj']
    2026-10-07T13:27:13.2139 | GPTQ | METRIC - time 0.21s
    2026-10-07T13:27:13.2160 | GPTQ | METRIC - error 274935.56
    2026-10-07T13:27:13.2186 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:27:13.2480 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.22.self_attn.o_proj']
    2026-10-07T13:27:14.0116 | GPTQ | METRIC - time 0.76s
    2026-10-07T13:27:14.0122 | GPTQ | METRIC - error 8635.87
    2026-10-07T13:27:14.0136 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:27:14.0228 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.22.mlp.gate_proj']
    2026-10-07T13:27:14.3204 | GPTQ | METRIC - time 0.30s
    2026-10-07T13:27:14.3246 | GPTQ | METRIC - error 50567.34
    2026-10-07T13:27:14.3275 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:27:14.3377 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.22.mlp.up_proj']
    2026-10-07T13:27:14.6477 | GPTQ | METRIC - time 0.31s
    2026-10-07T13:27:14.6498 | GPTQ | METRIC - error 65537.30
    2026-10-07T13:27:14.6517 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:27:14.6790 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.22.mlp.down_proj']
    2026-10-07T13:27:16.1098 | GPTQ | METRIC - time 1.42s
    2026-10-07T13:27:16.1103 | GPTQ | METRIC - error 56923.35
    2026-10-07T13:27:16.1107 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (24/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [10:38<00:00,  2.49s/it]
    (25/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [13:47<00:00,  3.23s/it]


    2026-10-07T13:51:55.2662 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.23.self_attn.q_proj']
    2026-10-07T13:51:55.7644 | GPTQ | METRIC - time 0.44s
    2026-10-07T13:51:55.7650 | GPTQ | METRIC - error 666665.25
    2026-10-07T13:51:55.7758 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:51:55.7909 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.23.self_attn.k_proj']
    2026-10-07T13:51:55.9603 | GPTQ | METRIC - time 0.17s
    2026-10-07T13:51:55.9608 | GPTQ | METRIC - error 278241.06
    2026-10-07T13:51:55.9613 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:51:55.9641 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.23.self_attn.v_proj']
    2026-10-07T13:51:56.0742 | GPTQ | METRIC - time 0.11s
    2026-10-07T13:51:56.0751 | GPTQ | METRIC - error 320370.03
    2026-10-07T13:51:56.0756 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:51:56.0886 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.23.self_attn.o_proj']
    2026-10-07T13:51:56.3483 | GPTQ | METRIC - time 0.26s
    2026-10-07T13:51:56.3487 | GPTQ | METRIC - error 11673.01
    2026-10-07T13:51:56.3491 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:51:56.3593 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.23.mlp.gate_proj']
    2026-10-07T13:51:56.5037 | GPTQ | METRIC - time 0.14s
    2026-10-07T13:51:56.5053 | GPTQ | METRIC - error 57711.89
    2026-10-07T13:51:56.5071 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:51:56.5092 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.23.mlp.up_proj']
    2026-10-07T13:51:56.6596 | GPTQ | METRIC - time 0.15s
    2026-10-07T13:51:56.6605 | GPTQ | METRIC - error 76466.31
    2026-10-07T13:51:56.6613 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T13:51:56.6713 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.23.mlp.down_proj']
    2026-10-07T13:51:57.6869 | GPTQ | METRIC - time 1.01s
    2026-10-07T13:51:57.6874 | GPTQ | METRIC - error 61795.71
    2026-10-07T13:51:57.6885 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (25/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [09:30<00:00,  2.23s/it]
    (26/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [14:16<00:00,  3.34s/it]


    2026-10-07T14:16:17.0219 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.24.self_attn.q_proj']
    2026-10-07T14:16:17.6496 | GPTQ | METRIC - time 0.52s
    2026-10-07T14:16:17.6503 | GPTQ | METRIC - error 1352537.75
    2026-10-07T14:16:17.6590 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:16:17.6789 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.24.self_attn.k_proj']
    2026-10-07T14:16:18.2318 | GPTQ | METRIC - time 0.55s
    2026-10-07T14:16:18.2323 | GPTQ | METRIC - error 468753.88
    2026-10-07T14:16:18.2327 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:16:18.2358 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.24.self_attn.v_proj']
    2026-10-07T14:16:18.3460 | GPTQ | METRIC - time 0.11s
    2026-10-07T14:16:18.3465 | GPTQ | METRIC - error 505463.06
    2026-10-07T14:16:18.3477 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:16:18.3570 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.24.self_attn.o_proj']
    2026-10-07T14:16:19.4910 | GPTQ | METRIC - time 1.13s
    2026-10-07T14:16:19.4955 | GPTQ | METRIC - error 12200.80
    2026-10-07T14:16:19.4961 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:16:19.5376 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.24.mlp.gate_proj']
    2026-10-07T14:16:19.7139 | GPTQ | METRIC - time 0.17s
    2026-10-07T14:16:19.7150 | GPTQ | METRIC - error 53749.32
    2026-10-07T14:16:19.7168 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:16:19.7194 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.24.mlp.up_proj']
    2026-10-07T14:16:19.8864 | GPTQ | METRIC - time 0.17s
    2026-10-07T14:16:19.8870 | GPTQ | METRIC - error 74395.20
    2026-10-07T14:16:19.8927 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:16:19.9351 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.24.mlp.down_proj']
    2026-10-07T14:16:21.1207 | GPTQ | METRIC - time 1.18s
    2026-10-07T14:16:21.1213 | GPTQ | METRIC - error 67061.87
    2026-10-07T14:16:21.1219 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (26/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [11:28<00:00,  2.69s/it]
    (27/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [10:21<00:00,  2.43s/it]


    2026-10-07T14:38:55.4859 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.25.self_attn.q_proj']
    2026-10-07T14:38:56.0163 | GPTQ | METRIC - time 0.46s
    2026-10-07T14:38:56.0183 | GPTQ | METRIC - error 1830119.00
    2026-10-07T14:38:56.0270 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:38:56.0501 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.25.self_attn.k_proj']
    2026-10-07T14:38:56.6401 | GPTQ | METRIC - time 0.59s
    2026-10-07T14:38:56.6406 | GPTQ | METRIC - error 558266.50
    2026-10-07T14:38:56.6442 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:38:56.6477 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.25.self_attn.v_proj']
    2026-10-07T14:38:56.7654 | GPTQ | METRIC - time 0.12s
    2026-10-07T14:38:56.7658 | GPTQ | METRIC - error 821098.69
    2026-10-07T14:38:56.7663 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:38:56.7792 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.25.self_attn.o_proj']
    2026-10-07T14:38:57.8308 | GPTQ | METRIC - time 1.05s
    2026-10-07T14:38:57.8314 | GPTQ | METRIC - error 14406.61
    2026-10-07T14:38:57.8342 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:38:57.8770 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.25.mlp.gate_proj']
    2026-10-07T14:38:58.0314 | GPTQ | METRIC - time 0.15s
    2026-10-07T14:38:58.0318 | GPTQ | METRIC - error 51401.43
    2026-10-07T14:38:58.0323 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:38:58.0348 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.25.mlp.up_proj']
    2026-10-07T14:38:58.1732 | GPTQ | METRIC - time 0.14s
    2026-10-07T14:38:58.1736 | GPTQ | METRIC - error 75130.36
    2026-10-07T14:38:58.1769 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T14:38:58.1872 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.25.mlp.down_proj']
    2026-10-07T14:38:59.2205 | GPTQ | METRIC - time 1.03s
    2026-10-07T14:38:59.2230 | GPTQ | METRIC - error 80521.09
    2026-10-07T14:38:59.2289 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (27/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [11:13<00:00,  2.63s/it]
    (28/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [12:04<00:00,  2.83s/it]


    2026-10-07T15:02:40.6947 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.26.self_attn.q_proj']
    2026-10-07T15:02:41.4081 | GPTQ | METRIC - time 0.62s
    2026-10-07T15:02:41.4087 | GPTQ | METRIC - error 2189580.50
    2026-10-07T15:02:41.4247 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:02:41.4475 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.26.self_attn.k_proj']
    2026-10-07T15:02:41.9268 | GPTQ | METRIC - time 0.48s
    2026-10-07T15:02:41.9272 | GPTQ | METRIC - error 560388.50
    2026-10-07T15:02:41.9297 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:02:41.9321 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.26.self_attn.v_proj']
    2026-10-07T15:02:42.0447 | GPTQ | METRIC - time 0.11s
    2026-10-07T15:02:42.0451 | GPTQ | METRIC - error 788641.00
    2026-10-07T15:02:42.0502 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:02:42.0618 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.26.self_attn.o_proj']
    2026-10-07T15:02:42.5442 | GPTQ | METRIC - time 0.48s
    2026-10-07T15:02:42.5446 | GPTQ | METRIC - error 53878.32
    2026-10-07T15:02:42.5451 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:02:42.5503 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.26.mlp.gate_proj']
    2026-10-07T15:02:42.6936 | GPTQ | METRIC - time 0.14s
    2026-10-07T15:02:42.6939 | GPTQ | METRIC - error 50987.39
    2026-10-07T15:02:42.6947 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:02:42.6975 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.26.mlp.up_proj']
    2026-10-07T15:02:42.8352 | GPTQ | METRIC - time 0.14s
    2026-10-07T15:02:42.8356 | GPTQ | METRIC - error 73345.95
    2026-10-07T15:02:42.8360 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:02:42.8469 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.26.mlp.down_proj']
    2026-10-07T15:02:43.5405 | GPTQ | METRIC - time 0.69s
    2026-10-07T15:02:43.5409 | GPTQ | METRIC - error 100981.05
    2026-10-07T15:02:43.5413 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (28/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [11:12<00:00,  2.63s/it]
    (29/29): Calibrating: 100%|██████████████████████████████████████████████████| 256/256 [09:43<00:00,  2.28s/it]


    2026-10-07T15:24:01.6220 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.27.self_attn.q_proj']
    2026-10-07T15:24:02.3369 | GPTQ | METRIC - time 0.58s
    2026-10-07T15:24:02.3386 | GPTQ | METRIC - error 1053135.00
    2026-10-07T15:24:02.3457 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:24:02.3629 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.27.self_attn.k_proj']
    2026-10-07T15:24:04.1822 | GPTQ | METRIC - time 1.82s
    2026-10-07T15:24:04.1828 | GPTQ | METRIC - error 464706.91
    2026-10-07T15:24:04.1837 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:24:04.1860 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.27.self_attn.v_proj']
    2026-10-07T15:24:04.3033 | GPTQ | METRIC - time 0.12s
    2026-10-07T15:24:04.3037 | GPTQ | METRIC - error 575670.75
    2026-10-07T15:24:04.3042 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:24:04.3127 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.27.self_attn.o_proj']
    2026-10-07T15:24:08.0260 | GPTQ | METRIC - time 3.71s
    2026-10-07T15:24:08.0313 | GPTQ | METRIC - error 32339.93
    2026-10-07T15:24:08.0328 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:24:08.0437 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.27.mlp.gate_proj']
    2026-10-07T15:24:08.4363 | GPTQ | METRIC - time 0.39s
    2026-10-07T15:24:08.4395 | GPTQ | METRIC - error 113188.14
    2026-10-07T15:24:08.4409 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:24:08.4477 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.27.mlp.up_proj']
    2026-10-07T15:24:08.7842 | GPTQ | METRIC - time 0.33s
    2026-10-07T15:24:08.7848 | GPTQ | METRIC - error 129753.32
    2026-10-07T15:24:08.7855 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb
    2026-10-07T15:24:08.7949 | compress_module_list | INFO - Quantizing 1 module(s): ['model.layers.27.mlp.down_proj']
    2026-10-07T15:24:10.6851 | GPTQ | METRIC - time 1.89s
    2026-10-07T15:24:10.6857 | GPTQ | METRIC - error 134818.08
    2026-10-07T15:24:10.6865 | GPTQ | METRIC - Accelerator 0 | usage: 16.12% | total memory: 12.7 Gb


    (29/29): Propagating: 100%|██████████████████████████████████████████████████| 256/256 [12:07<00:00,  2.84s/it]


    2026-10-07T15:36:19.5418 | finalize | INFO - Compression lifecycle finalized for 1 modifiers


    Compressing model: 100%|█████████████████████████████████████████████████████| 196/196 [00:22<00:00,  8.54it/s]


    2026-10-07T15:36:49.8193 | _retie_embeddings | INFO - Re-tied input/output embeddings; saving a single shared table.


    /Users/ayush.ganatra/model_inference/myenv/lib/python3.14/site-packages/transformers/modeling_utils.py:3426: UserWarning: Attempting to save a model with offloaded modules. Ensure that unallocated cpu memory exceeds the `shard_size` (50GB default)
      warnings.warn(



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]


    2026-10-07T15:36:54.9298 | resave_config | INFO - Resaved original config with patched fields to ./qwen-3-0-6b-comprssed/config.json


    Dispatching model: 100%|██████████████████████████████████████████████████| 427/427 [00:00<00:00, 16787.28it/s]





    Qwen3ForCausalLM(
      (model): Qwen3Model(
        (embed_tokens): Embedding(151936, 1024)
        (layers): ModuleList(
          (0-27): 28 x Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
              (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
              (act_fn): SiLUActivation()
            )
            (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
          )
        )
        (norm): Qwen3RMSNorm((1024,), eps=1e-06)
        (rotary_emb): Qwen3RotaryEmbedding()
      )
      (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
    )




```python
import lm_eval

MODELS = {
    "bf16": "./qwen-3-0-6b",
    "w4a16": "./qwen-3-0-6b-comprssed",
}

for name, path in MODELS.items():
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={path},dtype=bfloat16",
        tasks=["arc_easy", "hellaswag"],
        limit=1000,
        batch_size=8,
        device="mps",
    )
    print(f"\n=== {name} ===")
    for task, r in out["results"].items():
        print(task, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
```


    Loading weights:   0%|          | 0/310 [00:00<?, ?it/s]


    
      0%|                                                                                 | 0/1000 [00:00<?, ?it/s][A
     10%|██████▊                                                                | 96/1000 [00:00<00:00, 956.87it/s][A
     19%|█████████████▍                                                        | 192/1000 [00:00<00:00, 938.66it/s][A
     29%|████████████████████                                                  | 286/1000 [00:00<00:00, 934.68it/s][A
     38%|██████████████████████████▋                                           | 382/1000 [00:00<00:00, 943.97it/s][A
     48%|█████████████████████████████████▌                                    | 479/1000 [00:00<00:00, 952.49it/s][A
     57%|████████████████████████████████████████▎                             | 575/1000 [00:00<00:00, 941.40it/s][A
     67%|██████████████████████████████████████████████▉                       | 670/1000 [00:00<00:00, 937.02it/s][A
     77%|█████████████████████████████████████████████████████▌                | 766/1000 [00:00<00:00, 943.27it/s][A
     86%|████████████████████████████████████████████████████████████▍         | 863/1000 [00:00<00:00, 948.96it/s][A
    100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 947.14it/s][A
    
      0%|                                                                                 | 0/1000 [00:00<?, ?it/s][A
     26%|█████████████████▌                                                   | 255/1000 [00:00<00:00, 2545.09it/s][A
     51%|███████████████████████████████████▎                                 | 512/1000 [00:00<00:00, 2556.98it/s][A
    100%|████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 2578.72it/s][A
    
    Tokenizing inputs:   0%|                                                              | 0/7997 [00:00<?, ?it/s][A
    Tokenizing inputs:   8%|████                                              | 656/7997 [00:00<00:01, 6558.30it/s][A
    Tokenizing inputs:  18%|████████▋                                        | 1425/7997 [00:00<00:00, 7219.08it/s][A
    Tokenizing inputs:  27%|█████████████▏                                   | 2147/7997 [00:00<00:00, 7212.43it/s][A
    Tokenizing inputs:  37%|█████████████████▉                               | 2923/7997 [00:00<00:00, 7425.93it/s][A
    Tokenizing inputs:  46%|██████████████████████▋                          | 3701/7997 [00:00<00:00, 7553.37it/s][A
    Tokenizing inputs:  56%|███████████████████████████▎                     | 4457/7997 [00:00<00:00, 7334.14it/s][A
    Tokenizing inputs:  65%|███████████████████████████████▊                 | 5192/7997 [00:00<00:00, 7195.07it/s][A
    Tokenizing inputs:  74%|████████████████████████████████████▏            | 5913/7997 [00:00<00:00, 7173.64it/s][A
    Tokenizing inputs:  83%|████████████████████████████████████████▊        | 6668/7997 [00:00<00:00, 7286.19it/s][A
    Tokenizing inputs: 100%|█████████████████████████████████████████████████| 7997/7997 [00:01<00:00, 7232.59it/s][A
    
    Running loglikelihood requests:   0%|                                                 | 0/7997 [00:00<?, ?it/s][A
    Running loglikelihood requests:   0%|                                     | 1/7997 [01:34<210:28:27, 94.76s/it][A
    Running loglikelihood requests:   0%|                                      | 3/7997 [01:34<54:38:13, 24.61s/it][A
    Running loglikelihood requests:   0%|                                      | 9/7997 [01:46<16:26:42,  7.41s/it][A
    Running loglikelihood requests:   0%|                                      | 17/7997 [02:02<9:14:15,  4.17s/it][A
    Running loglikelihood requests:   0%|                                      | 25/7997 [02:17<6:57:09,  3.14s/it][A
    Running loglikelihood requests:   1%|▏                                     | 40/7997 [02:20<3:22:07,  1.52s/it][A
    Running loglikelihood requests:   1%|▏                                     | 47/7997 [02:31<3:21:56,  1.52s/it][A
    Running loglikelihood requests:   1%|▏                                     | 48/7997 [02:35<3:36:43,  1.64s/it][A
    Running loglikelihood requests:   1%|▎                                     | 56/7997 [02:37<2:39:25,  1.20s/it][A
    Running loglikelihood requests:   1%|▎                                     | 64/7997 [02:38<1:58:04,  1.12it/s][A
    Running loglikelihood requests:   1%|▎                                     | 69/7997 [02:38<1:33:54,  1.41it/s][A
    Running loglikelihood requests:   1%|▎                                     | 72/7997 [02:40<1:27:25,  1.51it/s][A
    Running loglikelihood requests:   1%|▍                                     | 80/7997 [02:41<1:03:01,  2.09it/s][A
    Running loglikelihood requests:   1%|▍                                     | 87/7997 [02:52<1:02:57,  2.09it/s][A
    Running loglikelihood requests:   1%|▍                                     | 88/7997 [02:57<2:13:21,  1.01s/it][A
    Running loglikelihood requests:   1%|▍                                     | 91/7997 [02:57<1:53:16,  1.16it/s][A
    Running loglikelihood requests:   1%|▍                                     | 96/7997 [03:02<1:54:44,  1.15it/s][A
    Running loglikelihood requests:   1%|▍                                    | 107/7997 [03:04<1:14:16,  1.77it/s][A
    Running loglikelihood requests:   1%|▌                                    | 115/7997 [03:11<1:27:07,  1.51it/s][A
    Running loglikelihood requests:   2%|▌                                    | 123/7997 [03:14<1:16:08,  1.72it/s][A
    Running loglikelihood requests:   2%|▋                                      | 131/7997 [03:15<59:22,  2.21it/s][A
    Running loglikelihood requests:   2%|▋                                      | 139/7997 [03:17<47:12,  2.77it/s][A
    Running loglikelihood requests:   2%|▋                                      | 147/7997 [03:18<38:43,  3.38it/s][A
    Running loglikelihood requests:   2%|▊                                      | 155/7997 [03:19<32:37,  4.01it/s][A
    Running loglikelihood requests:   2%|▊                                      | 163/7997 [03:21<30:19,  4.31it/s][A
    Running loglikelihood requests:   2%|▊                                      | 171/7997 [03:23<31:39,  4.12it/s][A
    Running loglikelihood requests:   2%|▊                                      | 179/7997 [03:25<31:33,  4.13it/s][A
    Running loglikelihood requests:   2%|▉                                      | 187/7997 [03:25<25:09,  5.17it/s][A
    Running loglikelihood requests:   2%|▉                                      | 195/7997 [03:26<23:17,  5.58it/s][A
    Running loglikelihood requests:   3%|▉                                      | 203/7997 [03:28<22:18,  5.82it/s][A
    Running loglikelihood requests:   3%|█                                      | 213/7997 [03:29<18:54,  6.86it/s][A
    Running loglikelihood requests:   3%|█                                      | 221/7997 [03:30<17:52,  7.25it/s][A
    Running loglikelihood requests:   3%|█                                      | 229/7997 [03:31<17:43,  7.31it/s][A
    Running loglikelihood requests:   3%|█▏                                     | 237/7997 [03:32<16:43,  7.74it/s][A
    Running loglikelihood requests:   3%|█▏                                     | 245/7997 [03:32<14:48,  8.73it/s][A
    Running loglikelihood requests:   3%|█▏                                     | 253/7997 [03:33<14:07,  9.14it/s][A
    Running loglikelihood requests:   3%|█▎                                     | 261/7997 [03:34<13:49,  9.32it/s][A
    Running loglikelihood requests:   3%|█▎                                     | 269/7997 [03:35<15:13,  8.46it/s][A
    Running loglikelihood requests:   3%|█▎                                     | 277/7997 [03:36<15:09,  8.49it/s][A
    Running loglikelihood requests:   4%|█▍                                     | 288/7997 [03:37<13:10,  9.76it/s][A
    Running loglikelihood requests:   4%|█▍                                     | 296/7997 [03:37<12:47, 10.03it/s][A
    Running loglikelihood requests:   4%|█▍                                     | 304/7997 [03:38<12:56,  9.90it/s][A
    Running loglikelihood requests:   4%|█▌                                     | 312/7997 [03:39<12:25, 10.31it/s][A
    Running loglikelihood requests:   4%|█▌                                     | 320/7997 [03:40<12:43, 10.05it/s][A
    Running loglikelihood requests:   4%|█▌                                     | 328/7997 [03:41<12:48,  9.98it/s][A
    Running loglikelihood requests:   4%|█▋                                     | 336/7997 [03:41<12:18, 10.37it/s][A
    Running loglikelihood requests:   4%|█▋                                     | 344/7997 [03:42<11:46, 10.83it/s][A
    Running loglikelihood requests:   4%|█▋                                     | 352/7997 [03:46<28:20,  4.50it/s][A
    Running loglikelihood requests:   5%|█▊                                     | 360/7997 [03:48<29:37,  4.30it/s][A
    Running loglikelihood requests:   5%|█▊                                     | 368/7997 [03:50<27:40,  4.59it/s][A
    Running loglikelihood requests:   5%|█▊                                     | 369/7997 [03:50<27:18,  4.65it/s][A
    Running loglikelihood requests:   5%|█▊                                     | 376/7997 [03:53<35:09,  3.61it/s][A
    Running loglikelihood requests:   5%|█▊                                     | 384/7997 [03:54<32:55,  3.85it/s][A
    Running loglikelihood requests:   5%|█▉                                     | 392/7997 [03:56<27:16,  4.65it/s][A
    Running loglikelihood requests:   5%|█▉                                     | 400/7997 [03:56<21:43,  5.83it/s][A
    Running loglikelihood requests:   5%|█▉                                     | 409/7997 [03:57<17:58,  7.03it/s][A
    Running loglikelihood requests:   5%|██                                     | 417/7997 [03:58<16:58,  7.44it/s][A
    Running loglikelihood requests:   5%|██                                     | 425/7997 [03:59<16:32,  7.63it/s][A
    Running loglikelihood requests:   5%|██                                     | 433/7997 [04:00<15:45,  8.00it/s][A
    Running loglikelihood requests:   6%|██▏                                    | 441/7997 [04:01<16:05,  7.83it/s][A
    Running loglikelihood requests:   6%|██▏                                    | 449/7997 [04:02<16:34,  7.59it/s][A
    Running loglikelihood requests:   6%|██▏                                    | 457/7997 [04:03<15:49,  7.94it/s][A
    Running loglikelihood requests:   6%|██▎                                    | 465/7997 [04:09<40:25,  3.11it/s][A
    Running loglikelihood requests:   6%|██▎                                    | 473/7997 [04:11<35:21,  3.55it/s][A
    Running loglikelihood requests:   6%|██▎                                    | 481/7997 [04:11<29:01,  4.32it/s][A
    Running loglikelihood requests:   6%|██▍                                    | 489/7997 [04:12<24:08,  5.18it/s][A
    Running loglikelihood requests:   6%|██▍                                    | 497/7997 [04:13<20:05,  6.22it/s][A
    Running loglikelihood requests:   6%|██▍                                    | 505/7997 [04:14<17:59,  6.94it/s][A
    Running loglikelihood requests:   6%|██▌                                    | 513/7997 [04:15<16:05,  7.75it/s][A
    Running loglikelihood requests:   7%|██▌                                    | 521/7997 [04:15<15:28,  8.05it/s][A
    Running loglikelihood requests:   7%|██▌                                    | 532/7997 [04:16<13:31,  9.20it/s][A
    Running loglikelihood requests:   7%|██▋                                    | 544/7997 [04:17<11:49, 10.50it/s][A
    Running loglikelihood requests:   7%|██▋                                    | 552/7997 [04:18<12:01, 10.32it/s][A
    Running loglikelihood requests:   7%|██▋                                    | 560/7997 [04:19<12:27,  9.94it/s][A
    Running loglikelihood requests:   7%|██▊                                    | 568/7997 [04:20<12:45,  9.70it/s][A
    Running loglikelihood requests:   7%|██▊                                    | 576/7997 [04:21<12:48,  9.66it/s][A
    Running loglikelihood requests:   7%|██▊                                    | 585/7997 [04:22<13:15,  9.32it/s][A
    Running loglikelihood requests:   7%|██▉                                    | 595/7997 [04:23<12:29,  9.87it/s][A
    Running loglikelihood requests:   8%|██▉                                    | 603/7997 [04:24<13:05,  9.42it/s][A
    Running loglikelihood requests:   8%|██▉                                    | 611/7997 [04:25<15:19,  8.04it/s][A
    Running loglikelihood requests:   8%|███                                    | 619/7997 [04:26<17:42,  6.94it/s][A
    Running loglikelihood requests:   8%|███                                    | 627/7997 [04:27<16:39,  7.38it/s][A
    Running loglikelihood requests:   8%|███                                    | 635/7997 [04:28<15:05,  8.13it/s][A
    Running loglikelihood requests:   8%|███▏                                   | 643/7997 [04:29<14:12,  8.63it/s][A
    Running loglikelihood requests:   8%|███▏                                   | 651/7997 [04:30<14:19,  8.54it/s][A
    Running loglikelihood requests:   8%|███▏                                   | 659/7997 [04:31<14:12,  8.61it/s][A
    Running loglikelihood requests:   8%|███▎                                   | 667/7997 [04:32<13:49,  8.83it/s][A
    Running loglikelihood requests:   8%|███                                  | 675/7997 [04:47<1:20:02,  1.52it/s][A
    Running loglikelihood requests:   9%|███▏                                 | 683/7997 [04:55<1:32:23,  1.32it/s][A
    Running loglikelihood requests:   9%|███▏                                 | 691/7997 [05:17<2:44:20,  1.35s/it][A
    Running loglikelihood requests:   9%|███▏                                 | 699/7997 [05:19<2:05:24,  1.03s/it][A
    Running loglikelihood requests:   9%|███▎                                 | 707/7997 [05:20<1:32:42,  1.31it/s][A
    Running loglikelihood requests:   9%|███▎                                 | 715/7997 [05:21<1:08:31,  1.77it/s][A
    Running loglikelihood requests:   9%|███▌                                   | 723/7997 [05:22<51:51,  2.34it/s][A
    Running loglikelihood requests:   9%|███▌                                   | 731/7997 [05:23<39:29,  3.07it/s][A
    Running loglikelihood requests:   9%|███▌                                   | 739/7997 [05:23<30:16,  4.00it/s][A
    Running loglikelihood requests:   9%|███▋                                   | 747/7997 [05:24<23:36,  5.12it/s][A
    Running loglikelihood requests:   9%|███▋                                   | 755/7997 [05:24<19:03,  6.33it/s][A
    Running loglikelihood requests:  10%|███▋                                   | 763/7997 [05:25<17:36,  6.84it/s][A
    Running loglikelihood requests:  10%|███▊                                   | 771/7997 [05:26<16:12,  7.43it/s][A
    Running loglikelihood requests:  10%|███▊                                   | 779/7997 [05:27<14:18,  8.40it/s][A
    Running loglikelihood requests:  10%|███▊                                   | 787/7997 [05:27<12:48,  9.38it/s][A
    Running loglikelihood requests:  10%|███▉                                   | 795/7997 [05:28<12:04,  9.95it/s][A
    Running loglikelihood requests:  10%|███▉                                   | 803/7997 [05:29<12:31,  9.57it/s][A
    Running loglikelihood requests:  10%|███▉                                   | 811/7997 [05:30<13:41,  8.75it/s][A
    Running loglikelihood requests:  10%|███▉                                   | 819/7997 [05:33<23:03,  5.19it/s][A
    Running loglikelihood requests:  10%|████                                   | 827/7997 [05:34<19:33,  6.11it/s][A
    Running loglikelihood requests:  10%|████                                   | 835/7997 [05:34<15:59,  7.46it/s][A
    Running loglikelihood requests:  11%|████                                   | 843/7997 [05:35<13:46,  8.65it/s][A
    Running loglikelihood requests:  11%|████▏                                  | 857/7997 [05:36<09:40, 12.30it/s][A
    Running loglikelihood requests:  11%|████▏                                  | 865/7997 [05:36<09:07, 13.02it/s][A
    Running loglikelihood requests:  11%|████▎                                  | 873/7997 [05:37<10:26, 11.37it/s][A
    Running loglikelihood requests:  11%|████▎                                  | 881/7997 [05:38<11:01, 10.75it/s][A
    Running loglikelihood requests:  11%|████▎                                  | 889/7997 [05:39<11:02, 10.74it/s][A
    Running loglikelihood requests:  11%|████▎                                  | 897/7997 [05:39<11:15, 10.51it/s][A
    Running loglikelihood requests:  11%|████▍                                  | 905/7997 [05:40<11:27, 10.32it/s][A
    Running loglikelihood requests:  11%|████▍                                  | 913/7997 [05:41<11:16, 10.47it/s][A
    Running loglikelihood requests:  12%|████▍                                  | 921/7997 [05:42<10:54, 10.82it/s][A
    Running loglikelihood requests:  12%|████▌                                  | 929/7997 [05:43<11:37, 10.14it/s][A
    Running loglikelihood requests:  12%|████▌                                  | 937/7997 [05:43<11:48,  9.97it/s][A
    Running loglikelihood requests:  12%|████▌                                  | 947/7997 [05:44<10:40, 11.00it/s][A
    Running loglikelihood requests:  12%|████▋                                  | 955/7997 [05:45<11:36, 10.11it/s][A
    Running loglikelihood requests:  12%|████▋                                  | 963/7997 [05:46<12:10,  9.63it/s][A
    Running loglikelihood requests:  12%|████▋                                  | 971/7997 [05:47<11:39, 10.04it/s][A
    Running loglikelihood requests:  12%|████▊                                  | 979/7997 [05:48<11:50,  9.88it/s][A
    Running loglikelihood requests:  12%|████▊                                  | 987/7997 [05:49<12:37,  9.25it/s][A
    Running loglikelihood requests:  12%|████▊                                  | 995/7997 [05:49<12:36,  9.26it/s][A
    Running loglikelihood requests:  13%|████▊                                 | 1003/7997 [05:50<12:46,  9.12it/s][A
    Running loglikelihood requests:  13%|████▊                                 | 1011/7997 [05:51<12:25,  9.37it/s][A
    Running loglikelihood requests:  13%|████▊                                 | 1019/7997 [05:52<12:07,  9.59it/s][A
    Running loglikelihood requests:  13%|████▉                                 | 1027/7997 [05:53<11:56,  9.73it/s][A
    Running loglikelihood requests:  13%|████▉                                 | 1037/7997 [05:53<10:58, 10.57it/s][A
    Running loglikelihood requests:  13%|████▉                                 | 1045/7997 [05:54<11:42,  9.90it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1053/7997 [05:55<12:09,  9.52it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1060/7997 [05:55<09:25, 12.27it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1063/7997 [05:56<11:58,  9.65it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1069/7997 [05:57<12:56,  8.92it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1075/7997 [05:57<09:45, 11.83it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1078/7997 [05:58<12:51,  8.96it/s][A
    Running loglikelihood requests:  14%|█████▏                                | 1085/7997 [06:05<50:18,  2.29it/s][A
    Running loglikelihood requests:  14%|█████▏                                | 1093/7997 [06:06<36:25,  3.16it/s][A
    Running loglikelihood requests:  14%|█████▏                                | 1101/7997 [06:06<27:16,  4.21it/s][A
    Running loglikelihood requests:  14%|█████▎                                | 1109/7997 [06:07<22:01,  5.21it/s][A
    Running loglikelihood requests:  14%|█████▎                                | 1117/7997 [06:08<18:40,  6.14it/s][A
    Running loglikelihood requests:  14%|█████▎                                | 1128/7997 [06:09<14:40,  7.80it/s][A
    Running loglikelihood requests:  14%|█████▍                                | 1136/7997 [06:10<13:40,  8.36it/s][A
    Running loglikelihood requests:  14%|█████▍                                | 1144/7997 [06:11<13:23,  8.53it/s][A
    Running loglikelihood requests:  14%|█████▍                                | 1152/7997 [06:11<12:46,  8.93it/s][A
    Running loglikelihood requests:  15%|█████▌                                | 1160/7997 [06:12<12:15,  9.29it/s][A
    Running loglikelihood requests:  15%|█████▌                                | 1168/7997 [06:13<11:44,  9.69it/s][A
    Running loglikelihood requests:  15%|█████▌                                | 1176/7997 [06:13<10:55, 10.40it/s][A
    Running loglikelihood requests:  15%|█████▋                                | 1184/7997 [06:14<10:52, 10.44it/s][A
    Running loglikelihood requests:  15%|█████▋                                | 1192/7997 [06:15<11:22,  9.97it/s][A
    Running loglikelihood requests:  15%|█████▋                                | 1200/7997 [06:16<10:46, 10.51it/s][A
    Running loglikelihood requests:  15%|█████▋                                | 1208/7997 [06:17<11:23,  9.94it/s][A
    Running loglikelihood requests:  15%|█████▊                                | 1216/7997 [06:18<11:42,  9.65it/s][A
    Running loglikelihood requests:  15%|█████▊                                | 1226/7997 [06:18<10:55, 10.33it/s][A
    Running loglikelihood requests:  15%|█████▉                                | 1237/7997 [06:20<10:49, 10.40it/s][A
    Running loglikelihood requests:  16%|█████▉                                | 1245/7997 [06:20<11:42,  9.61it/s][A
    Running loglikelihood requests:  16%|█████▉                                | 1253/7997 [06:21<11:27,  9.81it/s][A
    Running loglikelihood requests:  16%|█████▉                                | 1261/7997 [06:22<10:52, 10.33it/s][A
    Running loglikelihood requests:  16%|██████                                | 1269/7997 [06:22<09:59, 11.22it/s][A
    Running loglikelihood requests:  16%|██████                                | 1277/7997 [06:23<09:27, 11.85it/s][A
    Running loglikelihood requests:  16%|██████                                | 1285/7997 [06:24<09:37, 11.63it/s][A
    Running loglikelihood requests:  16%|██████▏                               | 1293/7997 [06:24<09:30, 11.75it/s][A
    Running loglikelihood requests:  16%|██████▏                               | 1301/7997 [06:25<09:39, 11.56it/s][A
    Running loglikelihood requests:  16%|██████▏                               | 1309/7997 [06:26<10:20, 10.78it/s][A
    Running loglikelihood requests:  17%|██████▎                               | 1320/7997 [06:27<09:01, 12.32it/s][A
    Running loglikelihood requests:  17%|██████▎                               | 1328/7997 [06:27<08:54, 12.47it/s][A
    Running loglikelihood requests:  17%|██████▎                               | 1339/7997 [06:28<07:33, 14.68it/s][A
    Running loglikelihood requests:  17%|██████▍                               | 1347/7997 [06:28<07:31, 14.74it/s][A
    Running loglikelihood requests:  17%|██████▍                               | 1355/7997 [06:29<09:36, 11.52it/s][A
    Running loglikelihood requests:  17%|██████▍                               | 1363/7997 [06:30<10:11, 10.86it/s][A
    Running loglikelihood requests:  17%|██████▌                               | 1371/7997 [06:31<10:55, 10.11it/s][A
    Running loglikelihood requests:  17%|██████▌                               | 1379/7997 [06:32<11:26,  9.65it/s][A
    Running loglikelihood requests:  17%|██████▌                               | 1387/7997 [06:33<11:07,  9.91it/s][A
    Running loglikelihood requests:  17%|██████▋                               | 1395/7997 [06:37<26:15,  4.19it/s][A
    Running loglikelihood requests:  18%|██████▋                               | 1403/7997 [06:40<27:42,  3.97it/s][A
    Running loglikelihood requests:  18%|██████▋                               | 1411/7997 [06:40<22:39,  4.84it/s][A
    Running loglikelihood requests:  18%|██████▋                               | 1419/7997 [06:41<19:11,  5.71it/s][A
    Running loglikelihood requests:  18%|██████▊                               | 1427/7997 [06:42<16:27,  6.65it/s][A
    Running loglikelihood requests:  18%|██████▊                               | 1435/7997 [06:43<14:06,  7.75it/s][A
    Running loglikelihood requests:  18%|██████▊                               | 1444/7997 [06:43<12:11,  8.95it/s][A
    Running loglikelihood requests:  18%|██████▉                               | 1452/7997 [06:44<11:48,  9.23it/s][A
    Running loglikelihood requests:  18%|██████▉                               | 1460/7997 [06:45<10:57,  9.94it/s][A
    Running loglikelihood requests:  18%|██████▉                               | 1468/7997 [06:45<10:16, 10.59it/s][A
    Running loglikelihood requests:  18%|███████                               | 1476/7997 [06:46<09:34, 11.35it/s][A
    Running loglikelihood requests:  19%|███████                               | 1484/7997 [06:47<09:00, 12.05it/s][A
    Running loglikelihood requests:  19%|███████                               | 1492/7997 [06:47<09:13, 11.75it/s][A
    Running loglikelihood requests:  19%|███████▏                              | 1500/7997 [06:48<09:07, 11.87it/s][A
    Running loglikelihood requests:  19%|███████▏                              | 1508/7997 [06:49<08:31, 12.68it/s][A
    Running loglikelihood requests:  19%|███████▏                              | 1516/7997 [06:49<08:01, 13.45it/s][A
    Running loglikelihood requests:  19%|███████▏                              | 1524/7997 [06:50<07:34, 14.25it/s][A
    Running loglikelihood requests:  19%|███████▎                              | 1532/7997 [06:50<07:18, 14.76it/s][A
    Running loglikelihood requests:  19%|███████▎                              | 1540/7997 [06:50<07:04, 15.21it/s][A
    Running loglikelihood requests:  19%|███████▎                              | 1548/7997 [06:51<07:22, 14.57it/s][A
    Running loglikelihood requests:  19%|███████▍                              | 1556/7997 [06:52<07:04, 15.16it/s][A
    Running loglikelihood requests:  20%|███████▍                              | 1567/7997 [06:52<06:15, 17.10it/s][A
    Running loglikelihood requests:  20%|███████▍                              | 1575/7997 [06:53<06:23, 16.75it/s][A
    Running loglikelihood requests:  20%|███████▌                              | 1583/7997 [06:53<06:25, 16.65it/s][A
    Running loglikelihood requests:  20%|███████▌                              | 1591/7997 [06:54<06:52, 15.53it/s][A
    Running loglikelihood requests:  20%|███████▌                              | 1599/7997 [06:54<06:48, 15.66it/s][A
    Running loglikelihood requests:  20%|███████▋                              | 1607/7997 [06:55<07:02, 15.14it/s][A
    Running loglikelihood requests:  20%|███████▋                              | 1615/7997 [06:55<07:09, 14.87it/s][A
    Running loglikelihood requests:  20%|███████▋                              | 1623/7997 [07:01<27:35,  3.85it/s][A
    Running loglikelihood requests:  20%|███████▊                              | 1631/7997 [07:04<31:07,  3.41it/s][A
    Running loglikelihood requests:  20%|███████▊                              | 1639/7997 [07:06<29:05,  3.64it/s][A
    Running loglikelihood requests:  21%|███████▊                              | 1647/7997 [07:07<23:09,  4.57it/s][A
    Running loglikelihood requests:  21%|███████▊                              | 1655/7997 [07:07<18:57,  5.58it/s][A
    Running loglikelihood requests:  21%|███████▉                              | 1665/7997 [07:08<14:52,  7.10it/s][A
    Running loglikelihood requests:  21%|███████▉                              | 1673/7997 [07:09<13:45,  7.66it/s][A
    Running loglikelihood requests:  21%|███████▉                              | 1681/7997 [07:09<12:31,  8.40it/s][A
    Running loglikelihood requests:  21%|████████                              | 1689/7997 [07:10<11:37,  9.05it/s][A
    Running loglikelihood requests:  21%|████████                              | 1697/7997 [07:11<10:51,  9.66it/s][A
    Running loglikelihood requests:  21%|████████                              | 1705/7997 [07:12<10:13, 10.26it/s][A
    Running loglikelihood requests:  21%|████████▏                             | 1713/7997 [07:12<09:51, 10.62it/s][A
    Running loglikelihood requests:  22%|████████▏                             | 1721/7997 [07:13<09:37, 10.88it/s][A
    Running loglikelihood requests:  22%|████████▏                             | 1729/7997 [07:14<09:09, 11.40it/s][A
    Running loglikelihood requests:  22%|████████▎                             | 1737/7997 [07:14<09:32, 10.93it/s][A
    Running loglikelihood requests:  22%|████████▎                             | 1745/7997 [07:15<09:17, 11.21it/s][A
    Running loglikelihood requests:  22%|████████▎                             | 1753/7997 [07:16<10:01, 10.39it/s][A
    Running loglikelihood requests:  22%|████████▎                             | 1761/7997 [07:17<09:40, 10.75it/s][A
    Running loglikelihood requests:  22%|████████▍                             | 1769/7997 [07:17<09:44, 10.65it/s][A
    Running loglikelihood requests:  22%|████████▍                             | 1777/7997 [07:18<09:24, 11.01it/s][A
    Running loglikelihood requests:  22%|████████▍                             | 1785/7997 [07:19<09:22, 11.05it/s][A
    Running loglikelihood requests:  22%|████████▍                             | 1787/7997 [07:19<09:00, 11.49it/s][A
    Running loglikelihood requests:  22%|████████▌                             | 1793/7997 [07:20<10:08, 10.20it/s][A
    Running loglikelihood requests:  23%|████████▌                             | 1801/7997 [07:20<09:27, 10.91it/s][A
    Running loglikelihood requests:  23%|████████▌                             | 1809/7997 [07:21<09:07, 11.29it/s][A
    Running loglikelihood requests:  23%|████████▋                             | 1817/7997 [07:22<08:45, 11.76it/s][A
    Running loglikelihood requests:  23%|████████▋                             | 1825/7997 [07:22<08:47, 11.71it/s][A
    Running loglikelihood requests:  23%|████████▋                             | 1832/7997 [07:22<06:41, 15.34it/s][A
    Running loglikelihood requests:  23%|████████▋                             | 1835/7997 [07:23<08:46, 11.71it/s][A
    Running loglikelihood requests:  23%|████████▋                             | 1841/7997 [07:24<09:37, 10.67it/s][A
    Running loglikelihood requests:  23%|████████▊                             | 1849/7997 [07:24<08:57, 11.43it/s][A
    Running loglikelihood requests:  23%|████████▊                             | 1857/7997 [07:25<08:33, 11.95it/s][A
    Running loglikelihood requests:  23%|████████▊                             | 1865/7997 [07:25<08:06, 12.59it/s][A
    Running loglikelihood requests:  23%|████████▉                             | 1873/7997 [07:26<08:03, 12.66it/s][A
    Running loglikelihood requests:  24%|████████▉                             | 1881/7997 [07:27<07:51, 12.97it/s][A
    Running loglikelihood requests:  24%|████████▉                             | 1889/7997 [07:27<07:50, 12.99it/s][A
    Running loglikelihood requests:  24%|█████████                             | 1897/7997 [07:28<08:45, 11.60it/s][A
    Running loglikelihood requests:  24%|█████████                             | 1905/7997 [07:29<08:29, 11.96it/s][A
    Running loglikelihood requests:  24%|█████████                             | 1913/7997 [07:29<08:45, 11.58it/s][A
    Running loglikelihood requests:  24%|█████████▏                            | 1921/7997 [07:30<08:40, 11.66it/s][A
    Running loglikelihood requests:  24%|█████████▏                            | 1929/7997 [07:31<08:42, 11.62it/s][A
    Running loglikelihood requests:  24%|█████████▏                            | 1937/7997 [07:31<08:21, 12.08it/s][A
    Running loglikelihood requests:  24%|█████████▏                            | 1945/7997 [07:32<08:26, 11.95it/s][A
    Running loglikelihood requests:  24%|█████████▎                            | 1953/7997 [07:33<08:20, 12.07it/s][A
    Running loglikelihood requests:  25%|█████████▎                            | 1961/7997 [07:33<08:07, 12.37it/s][A
    Running loglikelihood requests:  25%|█████████▎                            | 1969/7997 [07:34<08:00, 12.55it/s][A
    Running loglikelihood requests:  25%|█████████▍                            | 1977/7997 [07:35<08:10, 12.27it/s][A
    Running loglikelihood requests:  25%|█████████▍                            | 1985/7997 [07:36<08:44, 11.45it/s][A
    Running loglikelihood requests:  25%|█████████▍                            | 1993/7997 [07:36<09:17, 10.77it/s][A
    Running loglikelihood requests:  25%|█████████▌                            | 2001/7997 [07:37<09:17, 10.75it/s][A
    Running loglikelihood requests:  25%|█████████▌                            | 2009/7997 [07:38<09:12, 10.84it/s][A
    Running loglikelihood requests:  25%|█████████▌                            | 2017/7997 [07:38<08:51, 11.25it/s][A
    Running loglikelihood requests:  25%|█████████▌                            | 2025/7997 [07:39<09:00, 11.05it/s][A
    Running loglikelihood requests:  25%|█████████▋                            | 2033/7997 [07:40<08:50, 11.25it/s][A
    Running loglikelihood requests:  26%|█████████▋                            | 2041/7997 [07:41<09:16, 10.70it/s][A
    Running loglikelihood requests:  26%|█████████▋                            | 2049/7997 [07:41<09:05, 10.91it/s][A
    Running loglikelihood requests:  26%|█████████▊                            | 2057/7997 [07:42<08:51, 11.18it/s][A
    Running loglikelihood requests:  26%|█████████▊                            | 2065/7997 [07:43<08:31, 11.60it/s][A
    Running loglikelihood requests:  26%|█████████▊                            | 2073/7997 [07:43<08:26, 11.69it/s][A
    Running loglikelihood requests:  26%|█████████▉                            | 2081/7997 [07:44<07:53, 12.48it/s][A
    Running loglikelihood requests:  26%|█████████▉                            | 2089/7997 [07:45<08:10, 12.05it/s][A
    Running loglikelihood requests:  26%|█████████▉                            | 2097/7997 [07:45<08:11, 12.00it/s][A
    Running loglikelihood requests:  26%|██████████                            | 2105/7997 [07:46<08:00, 12.26it/s][A
    Running loglikelihood requests:  26%|██████████                            | 2113/7997 [07:47<08:08, 12.04it/s][A
    Running loglikelihood requests:  27%|██████████                            | 2121/7997 [07:47<08:11, 11.97it/s][A
    Running loglikelihood requests:  27%|██████████                            | 2129/7997 [07:48<08:02, 12.16it/s][A
    Running loglikelihood requests:  27%|██████████▏                           | 2137/7997 [07:49<07:50, 12.44it/s][A
    Running loglikelihood requests:  27%|██████████▏                           | 2145/7997 [07:49<07:56, 12.28it/s][A
    Running loglikelihood requests:  27%|██████████▏                           | 2153/7997 [07:56<30:56,  3.15it/s][A
    Running loglikelihood requests:  27%|██████████▎                           | 2161/7997 [07:57<24:56,  3.90it/s][A
    Running loglikelihood requests:  27%|██████████▎                           | 2169/7997 [07:58<19:54,  4.88it/s][A
    Running loglikelihood requests:  27%|██████████▎                           | 2177/7997 [07:58<15:57,  6.08it/s][A
    Running loglikelihood requests:  27%|██████████▍                           | 2185/7997 [07:59<13:39,  7.09it/s][A
    Running loglikelihood requests:  27%|██████████▍                           | 2193/7997 [08:00<11:52,  8.15it/s][A
    Running loglikelihood requests:  28%|██████████▍                           | 2201/7997 [08:00<10:38,  9.08it/s][A
    Running loglikelihood requests:  28%|██████████▍                           | 2209/7997 [08:01<09:23, 10.27it/s][A
    Running loglikelihood requests:  28%|██████████▌                           | 2218/7997 [08:01<08:14, 11.69it/s][A
    Running loglikelihood requests:  28%|██████████▌                           | 2226/7997 [08:02<08:36, 11.17it/s][A
    Running loglikelihood requests:  28%|██████████▌                           | 2234/7997 [08:03<08:18, 11.57it/s][A
    Running loglikelihood requests:  28%|██████████▋                           | 2242/7997 [08:04<08:18, 11.54it/s][A
    Running loglikelihood requests:  28%|██████████▋                           | 2250/7997 [08:04<08:05, 11.84it/s][A
    Running loglikelihood requests:  28%|██████████▋                           | 2258/7997 [08:05<07:48, 12.26it/s][A
    Running loglikelihood requests:  28%|██████████▊                           | 2266/7997 [08:05<07:46, 12.28it/s][A
    Running loglikelihood requests:  28%|██████████▊                           | 2274/7997 [08:06<07:36, 12.53it/s][A
    Running loglikelihood requests:  29%|██████████▊                           | 2282/7997 [08:07<07:33, 12.59it/s][A
    Running loglikelihood requests:  29%|██████████▉                           | 2290/7997 [08:07<07:37, 12.48it/s][A
    Running loglikelihood requests:  29%|██████████▉                           | 2298/7997 [08:08<07:41, 12.34it/s][A
    Running loglikelihood requests:  29%|██████████▉                           | 2306/7997 [08:09<07:48, 12.15it/s][A
    Running loglikelihood requests:  29%|██████████▉                           | 2314/7997 [08:09<07:19, 12.93it/s][A
    Running loglikelihood requests:  29%|███████████                           | 2322/7997 [08:10<07:23, 12.80it/s][A
    Running loglikelihood requests:  29%|███████████                           | 2330/7997 [08:10<07:27, 12.67it/s][A
    Running loglikelihood requests:  29%|███████████                           | 2338/7997 [08:11<07:33, 12.48it/s][A
    Running loglikelihood requests:  29%|███████████▏                          | 2346/7997 [08:12<07:40, 12.28it/s][A
    Running loglikelihood requests:  29%|███████████▏                          | 2357/7997 [08:12<06:53, 13.65it/s][A
    Running loglikelihood requests:  30%|███████████▏                          | 2365/7997 [08:13<06:54, 13.60it/s][A
    Running loglikelihood requests:  30%|███████████▎                          | 2373/7997 [08:14<06:52, 13.62it/s][A
    Running loglikelihood requests:  30%|███████████▎                          | 2382/7997 [08:14<06:44, 13.89it/s][A
    Running loglikelihood requests:  30%|███████████▎                          | 2393/7997 [08:15<06:07, 15.23it/s][A
    Running loglikelihood requests:  30%|███████████▍                          | 2401/7997 [08:15<06:25, 14.52it/s][A
    Running loglikelihood requests:  30%|███████████▍                          | 2409/7997 [08:16<07:03, 13.18it/s][A
    Running loglikelihood requests:  30%|███████████▍                          | 2417/7997 [08:17<07:26, 12.50it/s][A
    Running loglikelihood requests:  30%|███████████▌                          | 2425/7997 [08:18<07:10, 12.94it/s][A
    Running loglikelihood requests:  30%|███████████▌                          | 2433/7997 [08:18<06:56, 13.37it/s][A
    Running loglikelihood requests:  31%|███████████▌                          | 2441/7997 [08:19<07:51, 11.78it/s][A
    Running loglikelihood requests:  31%|███████████▋                          | 2449/7997 [08:20<08:24, 11.00it/s][A
    Running loglikelihood requests:  31%|███████████▋                          | 2457/7997 [08:20<08:14, 11.21it/s][A
    Running loglikelihood requests:  31%|███████████▋                          | 2465/7997 [08:21<07:56, 11.62it/s][A
    Running loglikelihood requests:  31%|███████████▊                          | 2473/7997 [08:22<08:01, 11.48it/s][A
    Running loglikelihood requests:  31%|███████████▊                          | 2481/7997 [08:22<07:56, 11.58it/s][A
    Running loglikelihood requests:  31%|███████████▊                          | 2489/7997 [08:23<07:50, 11.72it/s][A
    Running loglikelihood requests:  31%|███████████▊                          | 2497/7997 [08:24<07:40, 11.94it/s][A
    Running loglikelihood requests:  31%|███████████▉                          | 2505/7997 [08:24<07:24, 12.36it/s][A
    Running loglikelihood requests:  31%|███████████▉                          | 2513/7997 [08:25<07:10, 12.73it/s][A
    Running loglikelihood requests:  32%|███████████▉                          | 2521/7997 [08:26<07:00, 13.01it/s][A
    Running loglikelihood requests:  32%|████████████                          | 2529/7997 [08:26<06:54, 13.18it/s][A
    Running loglikelihood requests:  32%|████████████                          | 2541/7997 [08:27<05:51, 15.54it/s][A
    Running loglikelihood requests:  32%|████████████                          | 2550/7997 [08:27<06:09, 14.74it/s][A
    Running loglikelihood requests:  32%|████████████▏                         | 2558/7997 [08:28<06:26, 14.07it/s][A
    Running loglikelihood requests:  32%|████████████▏                         | 2572/7997 [08:29<05:26, 16.62it/s][A
    Running loglikelihood requests:  32%|████████████▎                         | 2580/7997 [08:29<05:45, 15.68it/s][A
    Running loglikelihood requests:  32%|████████████▎                         | 2588/7997 [08:30<06:00, 15.01it/s][A
    Running loglikelihood requests:  32%|████████████▎                         | 2596/7997 [08:30<06:11, 14.53it/s][A
    Running loglikelihood requests:  33%|████████████▎                         | 2604/7997 [08:31<06:39, 13.50it/s][A
    Running loglikelihood requests:  33%|████████████▍                         | 2612/7997 [08:32<06:35, 13.60it/s][A
    Running loglikelihood requests:  33%|████████████▍                         | 2620/7997 [08:32<06:30, 13.76it/s][A
    Running loglikelihood requests:  33%|████████████▍                         | 2628/7997 [08:33<06:39, 13.42it/s][A
    Running loglikelihood requests:  33%|████████████▌                         | 2636/7997 [08:34<06:54, 12.92it/s][A
    Running loglikelihood requests:  33%|████████████▌                         | 2644/7997 [08:34<06:59, 12.75it/s][A
    Running loglikelihood requests:  33%|████████████▌                         | 2652/7997 [08:35<07:01, 12.67it/s][A
    Running loglikelihood requests:  33%|████████████▋                         | 2660/7997 [08:39<17:43,  5.02it/s][A
    Running loglikelihood requests:  33%|████████████▋                         | 2668/7997 [08:40<16:37,  5.34it/s][A
    Running loglikelihood requests:  33%|████████████▋                         | 2676/7997 [08:41<13:41,  6.48it/s][A
    Running loglikelihood requests:  34%|████████████▊                         | 2684/7997 [08:41<11:44,  7.54it/s][A
    Running loglikelihood requests:  34%|████████████▊                         | 2692/7997 [08:42<10:00,  8.84it/s][A
    Running loglikelihood requests:  34%|████████████▊                         | 2700/7997 [08:42<08:59,  9.82it/s][A
    Running loglikelihood requests:  34%|████████████▊                         | 2708/7997 [08:43<08:11, 10.76it/s][A
    Running loglikelihood requests:  34%|████████████▉                         | 2716/7997 [08:44<07:50, 11.23it/s][A
    Running loglikelihood requests:  34%|████████████▉                         | 2724/7997 [08:44<07:27, 11.79it/s][A
    Running loglikelihood requests:  34%|████████████▉                         | 2732/7997 [08:45<06:54, 12.70it/s][A
    Running loglikelihood requests:  34%|█████████████                         | 2740/7997 [08:45<06:29, 13.49it/s][A
    Running loglikelihood requests:  34%|█████████████                         | 2748/7997 [08:46<06:11, 14.14it/s][A
    Running loglikelihood requests:  34%|█████████████                         | 2756/7997 [08:46<05:55, 14.73it/s][A
    Running loglikelihood requests:  35%|█████████████▏                        | 2764/7997 [08:47<05:45, 15.14it/s][A
    Running loglikelihood requests:  35%|█████████████▏                        | 2772/7997 [08:47<05:37, 15.49it/s][A
    Running loglikelihood requests:  35%|█████████████▏                        | 2780/7997 [08:48<05:29, 15.85it/s][A
    Running loglikelihood requests:  35%|█████████████▏                        | 2788/7997 [08:48<05:22, 16.13it/s][A
    Running loglikelihood requests:  35%|█████████████▎                        | 2796/7997 [08:49<05:23, 16.09it/s][A
    Running loglikelihood requests:  35%|█████████████▎                        | 2804/7997 [08:49<05:22, 16.13it/s][A
    Running loglikelihood requests:  35%|█████████████▎                        | 2812/7997 [08:50<06:22, 13.54it/s][A
    Running loglikelihood requests:  35%|█████████████▍                        | 2820/7997 [08:50<06:05, 14.15it/s][A
    Running loglikelihood requests:  35%|█████████████▍                        | 2828/7997 [08:51<06:03, 14.22it/s][A
    Running loglikelihood requests:  35%|█████████████▍                        | 2836/7997 [08:52<06:24, 13.43it/s][A
    Running loglikelihood requests:  36%|█████████████▌                        | 2844/7997 [08:52<06:00, 14.29it/s][A
    Running loglikelihood requests:  36%|█████████████▌                        | 2852/7997 [08:53<05:36, 15.31it/s][A
    Running loglikelihood requests:  36%|█████████████▌                        | 2860/7997 [08:53<05:50, 14.66it/s][A
    Running loglikelihood requests:  36%|█████████████▋                        | 2868/7997 [08:54<06:19, 13.50it/s][A
    Running loglikelihood requests:  36%|█████████████▋                        | 2876/7997 [08:55<06:22, 13.39it/s][A
    Running loglikelihood requests:  36%|█████████████▋                        | 2884/7997 [08:55<06:18, 13.50it/s][A
    Running loglikelihood requests:  36%|█████████████▋                        | 2892/7997 [08:56<05:59, 14.18it/s][A
    Running loglikelihood requests:  36%|█████████████▊                        | 2901/7997 [08:56<05:33, 15.30it/s][A
    Running loglikelihood requests:  36%|█████████████▊                        | 2911/7997 [08:57<05:38, 15.03it/s][A
    Running loglikelihood requests:  37%|█████████████▊                        | 2919/7997 [08:57<05:48, 14.56it/s][A
    Running loglikelihood requests:  37%|█████████████▉                        | 2927/7997 [08:58<05:32, 15.27it/s][A
    Running loglikelihood requests:  37%|█████████████▉                        | 2937/7997 [08:58<05:09, 16.33it/s][A
    Running loglikelihood requests:  37%|█████████████▉                        | 2945/7997 [08:59<05:16, 15.97it/s][A
    Running loglikelihood requests:  37%|██████████████                        | 2953/7997 [08:59<05:21, 15.68it/s][A
    Running loglikelihood requests:  37%|██████████████                        | 2961/7997 [09:00<05:14, 16.01it/s][A
    Running loglikelihood requests:  37%|██████████████                        | 2969/7997 [09:00<05:16, 15.88it/s][A
    Running loglikelihood requests:  37%|██████████████▏                       | 2977/7997 [09:01<05:04, 16.48it/s][A
    Running loglikelihood requests:  37%|██████████████▏                       | 2985/7997 [09:01<04:54, 17.00it/s][A
    Running loglikelihood requests:  37%|██████████████▏                       | 2993/7997 [09:02<04:49, 17.27it/s][A
    Running loglikelihood requests:  38%|██████████████▎                       | 3001/7997 [09:02<04:48, 17.34it/s][A
    Running loglikelihood requests:  38%|██████████████▎                       | 3009/7997 [09:03<04:45, 17.49it/s][A
    Running loglikelihood requests:  38%|██████████████▎                       | 3017/7997 [09:03<04:40, 17.75it/s][A
    Running loglikelihood requests:  38%|██████████████▎                       | 3025/7997 [09:04<04:40, 17.71it/s][A
    Running loglikelihood requests:  38%|██████████████▍                       | 3033/7997 [09:04<04:50, 17.10it/s][A
    Running loglikelihood requests:  38%|██████████████▍                       | 3041/7997 [09:04<04:39, 17.72it/s][A
    Running loglikelihood requests:  38%|██████████████▍                       | 3049/7997 [09:05<04:40, 17.64it/s][A
    Running loglikelihood requests:  38%|██████████████▌                       | 3057/7997 [09:05<04:40, 17.63it/s][A
    Running loglikelihood requests:  38%|██████████████▌                       | 3065/7997 [09:06<04:42, 17.48it/s][A
    Running loglikelihood requests:  38%|██████████████▌                       | 3073/7997 [09:06<04:40, 17.54it/s][A
    Running loglikelihood requests:  39%|██████████████▋                       | 3081/7997 [09:07<04:40, 17.52it/s][A
    Running loglikelihood requests:  39%|██████████████▋                       | 3089/7997 [09:07<04:37, 17.66it/s][A
    Running loglikelihood requests:  39%|██████████████▋                       | 3097/7997 [09:08<04:45, 17.17it/s][A
    Running loglikelihood requests:  39%|██████████████▊                       | 3105/7997 [09:08<04:47, 17.03it/s][A
    Running loglikelihood requests:  39%|██████████████▊                       | 3113/7997 [09:09<04:50, 16.83it/s][A
    Running loglikelihood requests:  39%|██████████████▊                       | 3121/7997 [09:09<05:26, 14.92it/s][A
    Running loglikelihood requests:  39%|██████████████▊                       | 3130/7997 [09:10<05:14, 15.47it/s][A
    Running loglikelihood requests:  39%|██████████████▉                       | 3146/7997 [09:10<03:55, 20.61it/s][A
    Running loglikelihood requests:  39%|██████████████▉                       | 3154/7997 [09:11<04:19, 18.65it/s][A
    Running loglikelihood requests:  40%|███████████████                       | 3162/7997 [09:11<04:27, 18.09it/s][A
    Running loglikelihood requests:  40%|███████████████                       | 3170/7997 [09:12<04:32, 17.71it/s][A
    Running loglikelihood requests:  40%|███████████████                       | 3178/7997 [09:12<04:39, 17.25it/s][A
    Running loglikelihood requests:  40%|███████████████▏                      | 3186/7997 [09:13<04:56, 16.22it/s][A
    Running loglikelihood requests:  40%|███████████████▏                      | 3194/7997 [09:13<04:53, 16.37it/s][A
    Running loglikelihood requests:  40%|███████████████▏                      | 3202/7997 [09:14<04:56, 16.18it/s][A
    Running loglikelihood requests:  40%|███████████████▎                      | 3210/7997 [09:14<04:58, 16.05it/s][A
    Running loglikelihood requests:  40%|███████████████▎                      | 3218/7997 [09:15<04:56, 16.09it/s][A
    Running loglikelihood requests:  40%|███████████████▎                      | 3226/7997 [09:15<05:01, 15.83it/s][A
    Running loglikelihood requests:  40%|███████████████▎                      | 3234/7997 [09:16<04:52, 16.26it/s][A
    Running loglikelihood requests:  41%|███████████████▍                      | 3242/7997 [09:16<04:47, 16.52it/s][A
    Running loglikelihood requests:  41%|███████████████▍                      | 3250/7997 [09:17<04:47, 16.52it/s][A
    Running loglikelihood requests:  41%|███████████████▍                      | 3258/7997 [09:17<04:50, 16.33it/s][A
    Running loglikelihood requests:  41%|███████████████▌                      | 3266/7997 [09:18<04:48, 16.40it/s][A
    Running loglikelihood requests:  41%|███████████████▌                      | 3274/7997 [09:18<04:41, 16.80it/s][A
    Running loglikelihood requests:  41%|███████████████▌                      | 3282/7997 [09:19<05:32, 14.19it/s][A
    Running loglikelihood requests:  41%|███████████████▌                      | 3285/7997 [09:19<05:08, 15.27it/s][A
    Running loglikelihood requests:  41%|███████████████▋                      | 3290/7997 [09:20<07:07, 11.02it/s][A
    Running loglikelihood requests:  41%|███████████████▋                      | 3294/7997 [09:20<06:06, 12.84it/s][A
    Running loglikelihood requests:  41%|███████████████▋                      | 3298/7997 [09:21<08:27,  9.25it/s][A
    Running loglikelihood requests:  41%|███████████████▋                      | 3306/7997 [09:22<07:36, 10.28it/s][A
    Running loglikelihood requests:  41%|███████████████▋                      | 3314/7997 [09:22<06:51, 11.39it/s][A
    Running loglikelihood requests:  42%|███████████████▊                      | 3322/7997 [09:23<06:34, 11.86it/s][A
    Running loglikelihood requests:  42%|███████████████▊                      | 3330/7997 [09:23<06:21, 12.23it/s][A
    Running loglikelihood requests:  42%|███████████████▊                      | 3339/7997 [09:24<05:43, 13.55it/s][A
    Running loglikelihood requests:  42%|███████████████▉                      | 3347/7997 [09:24<05:20, 14.50it/s][A
    Running loglikelihood requests:  42%|███████████████▉                      | 3356/7997 [09:25<04:57, 15.60it/s][A
    Running loglikelihood requests:  42%|███████████████▉                      | 3367/7997 [09:25<04:31, 17.07it/s][A
    Running loglikelihood requests:  42%|████████████████                      | 3375/7997 [09:26<04:26, 17.35it/s][A
    Running loglikelihood requests:  42%|████████████████                      | 3383/7997 [09:26<04:27, 17.28it/s][A
    Running loglikelihood requests:  42%|████████████████                      | 3391/7997 [09:27<04:28, 17.14it/s][A
    Running loglikelihood requests:  43%|████████████████▏                     | 3399/7997 [09:27<04:43, 16.25it/s][A
    Running loglikelihood requests:  43%|████████████████▏                     | 3407/7997 [09:28<04:42, 16.26it/s][A
    Running loglikelihood requests:  43%|████████████████▏                     | 3415/7997 [09:28<04:38, 16.44it/s][A
    Running loglikelihood requests:  43%|████████████████▎                     | 3423/7997 [09:29<04:32, 16.79it/s][A
    Running loglikelihood requests:  43%|████████████████▎                     | 3431/7997 [09:29<04:30, 16.89it/s][A
    Running loglikelihood requests:  43%|████████████████▎                     | 3439/7997 [09:30<04:32, 16.74it/s][A
    Running loglikelihood requests:  43%|████████████████▍                     | 3447/7997 [09:30<04:38, 16.36it/s][A
    Running loglikelihood requests:  43%|████████████████▍                     | 3455/7997 [09:31<04:58, 15.20it/s][A
    Running loglikelihood requests:  43%|████████████████▍                     | 3463/7997 [09:31<04:55, 15.36it/s][A
    Running loglikelihood requests:  43%|████████████████▍                     | 3471/7997 [09:32<04:58, 15.18it/s][A
    Running loglikelihood requests:  44%|████████████████▌                     | 3479/7997 [09:32<04:48, 15.64it/s][A
    Running loglikelihood requests:  44%|████████████████▌                     | 3487/7997 [09:33<04:35, 16.36it/s][A
    Running loglikelihood requests:  44%|████████████████▌                     | 3495/7997 [09:33<04:29, 16.71it/s][A
    Running loglikelihood requests:  44%|████████████████▋                     | 3503/7997 [09:34<04:19, 17.31it/s][A
    Running loglikelihood requests:  44%|████████████████▋                     | 3511/7997 [09:34<04:15, 17.53it/s][A
    Running loglikelihood requests:  44%|████████████████▋                     | 3519/7997 [09:35<04:10, 17.86it/s][A
    Running loglikelihood requests:  44%|████████████████▊                     | 3527/7997 [09:35<04:07, 18.05it/s][A
    Running loglikelihood requests:  44%|████████████████▊                     | 3535/7997 [09:35<04:07, 18.01it/s][A
    Running loglikelihood requests:  44%|████████████████▊                     | 3543/7997 [09:36<04:32, 16.37it/s][A
    Running loglikelihood requests:  44%|████████████████▉                     | 3552/7997 [09:36<04:12, 17.63it/s][A
    Running loglikelihood requests:  45%|████████████████▉                     | 3562/7997 [09:37<03:52, 19.11it/s][A
    Running loglikelihood requests:  45%|████████████████▉                     | 3570/7997 [09:37<03:55, 18.81it/s][A
    Running loglikelihood requests:  45%|█████████████████                     | 3578/7997 [09:38<03:58, 18.54it/s][A
    Running loglikelihood requests:  45%|█████████████████                     | 3587/7997 [09:38<03:48, 19.32it/s][A
    Running loglikelihood requests:  45%|█████████████████                     | 3595/7997 [09:39<04:30, 16.26it/s][A
    Running loglikelihood requests:  45%|█████████████████▏                    | 3607/7997 [09:39<04:10, 17.51it/s][A
    Running loglikelihood requests:  45%|█████████████████▏                    | 3616/7997 [09:40<04:15, 17.17it/s][A
    Running loglikelihood requests:  45%|█████████████████▏                    | 3624/7997 [09:41<04:19, 16.82it/s][A
    Running loglikelihood requests:  45%|█████████████████▎                    | 3632/7997 [09:41<04:18, 16.87it/s][A
    Running loglikelihood requests:  46%|█████████████████▎                    | 3640/7997 [09:42<04:29, 16.19it/s][A
    Running loglikelihood requests:  46%|█████████████████▎                    | 3648/7997 [09:42<04:28, 16.19it/s][A
    Running loglikelihood requests:  46%|█████████████████▎                    | 3656/7997 [09:43<04:26, 16.27it/s][A
    Running loglikelihood requests:  46%|█████████████████▍                    | 3664/7997 [09:43<04:25, 16.35it/s][A
    Running loglikelihood requests:  46%|█████████████████▍                    | 3672/7997 [09:44<04:24, 16.33it/s][A
    Running loglikelihood requests:  46%|█████████████████▍                    | 3680/7997 [09:44<04:16, 16.81it/s][A
    Running loglikelihood requests:  46%|█████████████████▌                    | 3688/7997 [09:45<04:29, 16.00it/s][A
    Running loglikelihood requests:  46%|█████████████████▌                    | 3696/7997 [09:45<04:52, 14.70it/s][A
    Running loglikelihood requests:  46%|█████████████████▌                    | 3704/7997 [09:46<04:56, 14.50it/s][A
    Running loglikelihood requests:  46%|█████████████████▋                    | 3712/7997 [09:46<04:52, 14.64it/s][A
    Running loglikelihood requests:  47%|█████████████████▋                    | 3720/7997 [09:47<04:41, 15.17it/s][A
    Running loglikelihood requests:  47%|█████████████████▋                    | 3728/7997 [09:47<04:27, 15.95it/s][A
    Running loglikelihood requests:  47%|█████████████████▊                    | 3736/7997 [09:48<04:16, 16.62it/s][A
    Running loglikelihood requests:  47%|█████████████████▊                    | 3744/7997 [09:48<04:17, 16.51it/s][A
    Running loglikelihood requests:  47%|█████████████████▊                    | 3752/7997 [09:49<04:08, 17.05it/s][A
    Running loglikelihood requests:  47%|█████████████████▊                    | 3760/7997 [09:49<04:06, 17.20it/s][A
    Running loglikelihood requests:  47%|█████████████████▉                    | 3768/7997 [09:50<04:24, 15.99it/s][A
    Running loglikelihood requests:  47%|█████████████████▉                    | 3776/7997 [09:50<04:15, 16.51it/s][A
    Running loglikelihood requests:  47%|█████████████████▉                    | 3784/7997 [09:50<04:06, 17.08it/s][A
    Running loglikelihood requests:  47%|██████████████████                    | 3792/7997 [09:51<04:15, 16.45it/s][A
    Running loglikelihood requests:  48%|██████████████████                    | 3806/7997 [09:51<03:29, 20.01it/s][A
    Running loglikelihood requests:  48%|██████████████████▏                   | 3815/7997 [09:52<03:30, 19.88it/s][A
    Running loglikelihood requests:  48%|██████████████████▏                   | 3826/7997 [09:52<03:13, 21.53it/s][A
    Running loglikelihood requests:  48%|██████████████████▏                   | 3836/7997 [09:53<03:35, 19.34it/s][A
    Running loglikelihood requests:  48%|██████████████████▎                   | 3844/7997 [09:53<03:35, 19.26it/s][A
    Running loglikelihood requests:  48%|██████████████████▎                   | 3852/7997 [09:54<03:45, 18.37it/s][A
    Running loglikelihood requests:  48%|██████████████████▎                   | 3860/7997 [09:54<03:46, 18.29it/s][A
    Running loglikelihood requests:  48%|██████████████████▍                   | 3868/7997 [09:55<03:53, 17.69it/s][A
    Running loglikelihood requests:  48%|██████████████████▍                   | 3877/7997 [09:55<03:39, 18.75it/s][A
    Running loglikelihood requests:  49%|██████████████████▍                   | 3885/7997 [09:56<03:37, 18.91it/s][A
    Running loglikelihood requests:  49%|██████████████████▍                   | 3893/7997 [09:56<03:40, 18.64it/s][A
    Running loglikelihood requests:  49%|██████████████████▌                   | 3901/7997 [09:57<03:44, 18.21it/s][A
    Running loglikelihood requests:  49%|██████████████████▌                   | 3909/7997 [09:57<03:45, 18.11it/s][A
    Running loglikelihood requests:  49%|██████████████████▌                   | 3917/7997 [09:57<03:43, 18.26it/s][A
    Running loglikelihood requests:  49%|██████████████████▋                   | 3925/7997 [09:58<03:48, 17.85it/s][A
    Running loglikelihood requests:  49%|██████████████████▋                   | 3933/7997 [09:58<03:46, 17.97it/s][A
    Running loglikelihood requests:  49%|██████████████████▋                   | 3941/7997 [09:59<03:42, 18.21it/s][A
    Running loglikelihood requests:  49%|██████████████████▊                   | 3949/7997 [09:59<03:46, 17.90it/s][A
    Running loglikelihood requests:  49%|██████████████████▊                   | 3957/7997 [10:00<03:41, 18.20it/s][A
    Running loglikelihood requests:  50%|██████████████████▊                   | 3965/7997 [10:00<03:51, 17.40it/s][A
    Running loglikelihood requests:  50%|██████████████████▉                   | 3973/7997 [10:01<03:46, 17.79it/s][A
    Running loglikelihood requests:  50%|██████████████████▉                   | 3981/7997 [10:01<04:03, 16.52it/s][A
    Running loglikelihood requests:  50%|██████████████████▉                   | 3989/7997 [10:02<03:52, 17.23it/s][A
    Running loglikelihood requests:  50%|██████████████████▉                   | 3997/7997 [10:02<03:52, 17.24it/s][A
    Running loglikelihood requests:  50%|███████████████████                   | 4005/7997 [10:03<03:52, 17.20it/s][A
    Running loglikelihood requests:  50%|███████████████████                   | 4013/7997 [10:03<03:56, 16.87it/s][A
    Running loglikelihood requests:  50%|███████████████████                   | 4023/7997 [10:03<03:31, 18.77it/s][A
    Running loglikelihood requests:  50%|███████████████████▏                  | 4031/7997 [10:04<03:34, 18.45it/s][A
    Running loglikelihood requests:  51%|███████████████████▏                  | 4039/7997 [10:04<03:34, 18.46it/s][A
    Running loglikelihood requests:  51%|███████████████████▏                  | 4049/7997 [10:05<03:20, 19.69it/s][A
    Running loglikelihood requests:  51%|███████████████████▎                  | 4057/7997 [10:05<03:22, 19.50it/s][A
    Running loglikelihood requests:  51%|███████████████████▎                  | 4065/7997 [10:06<03:24, 19.27it/s][A
    Running loglikelihood requests:  51%|███████████████████▎                  | 4073/7997 [10:06<03:27, 18.92it/s][A
    Running loglikelihood requests:  51%|███████████████████▍                  | 4081/7997 [10:06<03:25, 19.09it/s][A
    Running loglikelihood requests:  51%|███████████████████▍                  | 4089/7997 [10:07<03:28, 18.73it/s][A
    Running loglikelihood requests:  51%|███████████████████▍                  | 4097/7997 [10:07<03:29, 18.59it/s][A
    Running loglikelihood requests:  51%|███████████████████▌                  | 4105/7997 [10:08<03:28, 18.65it/s][A
    Running loglikelihood requests:  51%|███████████████████▌                  | 4113/7997 [10:08<03:25, 18.87it/s][A
    Running loglikelihood requests:  52%|███████████████████▌                  | 4121/7997 [10:09<03:24, 18.92it/s][A
    Running loglikelihood requests:  52%|███████████████████▌                  | 4129/7997 [10:09<03:30, 18.38it/s][A
    Running loglikelihood requests:  52%|███████████████████▋                  | 4137/7997 [10:10<03:30, 18.35it/s][A
    Running loglikelihood requests:  52%|███████████████████▋                  | 4145/7997 [10:10<03:43, 17.23it/s][A
    Running loglikelihood requests:  52%|███████████████████▋                  | 4153/7997 [10:11<04:02, 15.84it/s][A
    Running loglikelihood requests:  52%|███████████████████▊                  | 4161/7997 [10:11<03:53, 16.44it/s][A
    Running loglikelihood requests:  52%|███████████████████▊                  | 4169/7997 [10:12<03:47, 16.81it/s][A
    Running loglikelihood requests:  52%|███████████████████▊                  | 4177/7997 [10:12<03:50, 16.58it/s][A
    Running loglikelihood requests:  52%|███████████████████▉                  | 4185/7997 [10:13<03:45, 16.87it/s][A
    Running loglikelihood requests:  52%|███████████████████▉                  | 4193/7997 [10:13<04:06, 15.41it/s][A
    Running loglikelihood requests:  53%|███████████████████▉                  | 4201/7997 [10:14<03:52, 16.36it/s][A
    Running loglikelihood requests:  53%|████████████████████                  | 4210/7997 [10:14<03:45, 16.79it/s][A
    Running loglikelihood requests:  53%|████████████████████                  | 4218/7997 [10:15<03:44, 16.85it/s][A
    Running loglikelihood requests:  53%|████████████████████                  | 4226/7997 [10:15<03:59, 15.75it/s][A
    Running loglikelihood requests:  53%|████████████████████▏                 | 4237/7997 [10:16<03:27, 18.13it/s][A
    Running loglikelihood requests:  53%|████████████████████▏                 | 4245/7997 [10:16<04:09, 15.03it/s][A
    Running loglikelihood requests:  53%|████████████████████▏                 | 4253/7997 [10:17<04:20, 14.36it/s][A
    Running loglikelihood requests:  53%|████████████████████▏                 | 4261/7997 [10:17<04:04, 15.27it/s][A
    Running loglikelihood requests:  53%|████████████████████▎                 | 4271/7997 [10:18<03:33, 17.46it/s][A
    Running loglikelihood requests:  54%|████████████████████▎                 | 4279/7997 [10:18<03:30, 17.63it/s][A
    Running loglikelihood requests:  54%|████████████████████▎                 | 4287/7997 [10:19<03:55, 15.78it/s][A
    Running loglikelihood requests:  54%|████████████████████▍                 | 4295/7997 [10:20<04:15, 14.46it/s][A
    Running loglikelihood requests:  54%|████████████████████▍                 | 4303/7997 [10:20<04:14, 14.50it/s][A
    Running loglikelihood requests:  54%|████████████████████▍                 | 4311/7997 [10:21<04:07, 14.86it/s][A
    Running loglikelihood requests:  54%|████████████████████▌                 | 4319/7997 [10:21<03:56, 15.54it/s][A
    Running loglikelihood requests:  54%|████████████████████▌                 | 4327/7997 [10:22<04:01, 15.22it/s][A
    Running loglikelihood requests:  54%|████████████████████▌                 | 4335/7997 [10:22<04:04, 14.99it/s][A
    Running loglikelihood requests:  54%|████████████████████▋                 | 4343/7997 [10:23<03:57, 15.42it/s][A
    Running loglikelihood requests:  54%|████████████████████▋                 | 4351/7997 [10:23<03:44, 16.24it/s][A
    Running loglikelihood requests:  55%|████████████████████▋                 | 4359/7997 [10:24<03:39, 16.55it/s][A
    Running loglikelihood requests:  55%|████████████████████▊                 | 4367/7997 [10:24<03:39, 16.53it/s][A
    Running loglikelihood requests:  55%|████████████████████▊                 | 4375/7997 [10:24<03:28, 17.37it/s][A
    Running loglikelihood requests:  55%|████████████████████▊                 | 4383/7997 [10:25<03:22, 17.88it/s][A
    Running loglikelihood requests:  55%|████████████████████▊                 | 4391/7997 [10:25<03:19, 18.07it/s][A
    Running loglikelihood requests:  55%|████████████████████▉                 | 4400/7997 [10:26<03:10, 18.86it/s][A
    Running loglikelihood requests:  55%|████████████████████▉                 | 4408/7997 [10:26<03:10, 18.80it/s][A
    Running loglikelihood requests:  55%|████████████████████▉                 | 4416/7997 [10:27<03:15, 18.36it/s][A
    Running loglikelihood requests:  55%|█████████████████████                 | 4424/7997 [10:27<03:14, 18.41it/s][A
    Running loglikelihood requests:  55%|█████████████████████                 | 4432/7997 [10:28<03:15, 18.24it/s][A
    Running loglikelihood requests:  56%|█████████████████████                 | 4440/7997 [10:28<03:11, 18.53it/s][A
    Running loglikelihood requests:  56%|█████████████████████▏                | 4450/7997 [10:28<02:55, 20.16it/s][A
    Running loglikelihood requests:  56%|█████████████████████▏                | 4462/7997 [10:29<02:37, 22.43it/s][A
    Running loglikelihood requests:  56%|█████████████████████▏                | 4470/7997 [10:29<02:47, 21.11it/s][A
    Running loglikelihood requests:  56%|█████████████████████▎                | 4478/7997 [10:30<02:51, 20.55it/s][A
    Running loglikelihood requests:  56%|█████████████████████▎                | 4486/7997 [10:30<03:24, 17.16it/s][A
    Running loglikelihood requests:  56%|█████████████████████▎                | 4494/7997 [10:31<03:22, 17.34it/s][A
    Running loglikelihood requests:  56%|█████████████████████▍                | 4502/7997 [10:31<03:22, 17.23it/s][A
    Running loglikelihood requests:  56%|█████████████████████▍                | 4510/7997 [10:32<03:29, 16.61it/s][A
    Running loglikelihood requests:  56%|█████████████████████▍                | 4518/7997 [10:32<03:24, 17.04it/s][A
    Running loglikelihood requests:  57%|█████████████████████▌                | 4526/7997 [10:33<03:20, 17.32it/s][A
    Running loglikelihood requests:  57%|█████████████████████▌                | 4534/7997 [10:33<03:17, 17.54it/s][A
    Running loglikelihood requests:  57%|█████████████████████▌                | 4542/7997 [10:34<03:16, 17.58it/s][A
    Running loglikelihood requests:  57%|█████████████████████▌                | 4550/7997 [10:34<03:24, 16.85it/s][A
    Running loglikelihood requests:  57%|█████████████████████▋                | 4558/7997 [10:34<03:16, 17.53it/s][A
    Running loglikelihood requests:  57%|█████████████████████▋                | 4566/7997 [10:35<03:15, 17.57it/s][A
    Running loglikelihood requests:  57%|█████████████████████▋                | 4574/7997 [10:35<03:06, 18.35it/s][A
    Running loglikelihood requests:  57%|█████████████████████▊                | 4582/7997 [10:36<02:58, 19.09it/s][A
    Running loglikelihood requests:  57%|█████████████████████▊                | 4590/7997 [10:36<02:55, 19.46it/s][A
    Running loglikelihood requests:  57%|█████████████████████▊                | 4598/7997 [10:36<02:52, 19.66it/s][A
    Running loglikelihood requests:  58%|█████████████████████▉                | 4609/7997 [10:37<02:32, 22.22it/s][A
    Running loglikelihood requests:  58%|█████████████████████▉                | 4619/7997 [10:37<02:23, 23.48it/s][A
    Running loglikelihood requests:  58%|█████████████████████▉                | 4627/7997 [10:38<02:30, 22.43it/s][A
    Running loglikelihood requests:  58%|██████████████████████                | 4636/7997 [10:38<02:26, 23.02it/s][A
    Running loglikelihood requests:  58%|██████████████████████                | 4647/7997 [10:38<02:16, 24.47it/s][A
    Running loglikelihood requests:  58%|██████████████████████▏               | 4658/7997 [10:39<02:50, 19.55it/s][A
    Running loglikelihood requests:  58%|██████████████████████▏               | 4666/7997 [10:40<02:58, 18.63it/s][A
    Running loglikelihood requests:  58%|██████████████████████▏               | 4678/7997 [10:40<02:33, 21.61it/s][A
    Running loglikelihood requests:  59%|██████████████████████▎               | 4686/7997 [10:41<03:00, 18.39it/s][A
    Running loglikelihood requests:  59%|██████████████████████▎               | 4697/7997 [10:41<02:49, 19.46it/s][A
    Running loglikelihood requests:  59%|██████████████████████▎               | 4705/7997 [10:42<02:53, 18.93it/s][A
    Running loglikelihood requests:  59%|██████████████████████▍               | 4713/7997 [10:42<02:54, 18.77it/s][A
    Running loglikelihood requests:  59%|██████████████████████▍               | 4721/7997 [10:43<03:00, 18.10it/s][A
    Running loglikelihood requests:  59%|██████████████████████▍               | 4729/7997 [10:43<02:55, 18.58it/s][A
    Running loglikelihood requests:  59%|██████████████████████▌               | 4737/7997 [10:44<03:15, 16.66it/s][A
    Running loglikelihood requests:  59%|██████████████████████▌               | 4745/7997 [10:44<02:59, 18.10it/s][A
    Running loglikelihood requests:  59%|██████████████████████▌               | 4753/7997 [10:50<14:58,  3.61it/s][A
    Running loglikelihood requests:  60%|██████████████████████▌               | 4761/7997 [10:54<17:02,  3.17it/s][A
    Running loglikelihood requests:  60%|██████████████████████▋               | 4769/7997 [10:54<13:07,  4.10it/s][A
    Running loglikelihood requests:  60%|██████████████████████▋               | 4777/7997 [10:55<09:57,  5.39it/s][A
    Running loglikelihood requests:  60%|██████████████████████▋               | 4785/7997 [10:55<07:42,  6.95it/s][A
    Running loglikelihood requests:  60%|██████████████████████▊               | 4794/7997 [10:55<05:49,  9.17it/s][A
    Running loglikelihood requests:  60%|██████████████████████▊               | 4806/7997 [10:56<04:19, 12.30it/s][A
    Running loglikelihood requests:  60%|██████████████████████▉               | 4814/7997 [10:56<03:49, 13.88it/s][A
    Running loglikelihood requests:  60%|██████████████████████▉               | 4824/7997 [10:57<03:18, 15.95it/s][A
    Running loglikelihood requests:  60%|██████████████████████▉               | 4832/7997 [10:57<03:10, 16.61it/s][A
    Running loglikelihood requests:  61%|██████████████████████▉               | 4840/7997 [10:57<03:02, 17.27it/s][A
    Running loglikelihood requests:  61%|███████████████████████               | 4848/7997 [10:58<02:45, 19.08it/s][A
    Running loglikelihood requests:  61%|███████████████████████               | 4856/7997 [10:58<02:33, 20.49it/s][A
    Running loglikelihood requests:  61%|███████████████████████▏              | 4868/7997 [10:58<02:11, 23.84it/s][A
    Running loglikelihood requests:  61%|███████████████████████▏              | 4880/7997 [10:59<01:55, 27.10it/s][A
    Running loglikelihood requests:  61%|███████████████████████▏              | 4888/7997 [10:59<01:58, 26.32it/s][A
    Running loglikelihood requests:  61%|███████████████████████▎              | 4896/7997 [10:59<02:02, 25.30it/s][A
    Running loglikelihood requests:  61%|███████████████████████▎              | 4904/7997 [11:00<02:25, 21.26it/s][A
    Running loglikelihood requests:  61%|███████████████████████▎              | 4912/7997 [11:00<02:23, 21.56it/s][A
    Running loglikelihood requests:  62%|███████████████████████▍              | 4920/7997 [11:01<02:21, 21.73it/s][A
    Running loglikelihood requests:  62%|███████████████████████▍              | 4928/7997 [11:01<02:15, 22.67it/s][A
    Running loglikelihood requests:  62%|███████████████████████▍              | 4936/7997 [11:01<02:13, 23.01it/s][A
    Running loglikelihood requests:  62%|███████████████████████▍              | 4944/7997 [11:02<02:17, 22.16it/s][A
    Running loglikelihood requests:  62%|███████████████████████▌              | 4952/7997 [11:02<02:14, 22.65it/s][A
    Running loglikelihood requests:  62%|███████████████████████▌              | 4960/7997 [11:02<02:17, 22.02it/s][A
    Running loglikelihood requests:  62%|███████████████████████▌              | 4971/7997 [11:03<02:06, 23.90it/s][A
    Running loglikelihood requests:  62%|███████████████████████▋              | 4979/7997 [11:03<02:04, 24.25it/s][A
    Running loglikelihood requests:  62%|███████████████████████▋              | 4989/7997 [11:03<01:54, 26.17it/s][A
    Running loglikelihood requests:  62%|███████████████████████▋              | 4997/7997 [11:04<02:02, 24.58it/s][A
    Running loglikelihood requests:  63%|███████████████████████▊              | 5009/7997 [11:04<01:51, 26.82it/s][A
    Running loglikelihood requests:  63%|███████████████████████▊              | 5019/7997 [11:05<01:45, 28.10it/s][A
    Running loglikelihood requests:  63%|███████████████████████▉              | 5027/7997 [11:05<01:52, 26.44it/s][A
    Running loglikelihood requests:  63%|███████████████████████▉              | 5035/7997 [11:05<01:52, 26.27it/s][A
    Running loglikelihood requests:  63%|███████████████████████▉              | 5043/7997 [11:06<01:55, 25.68it/s][A
    Running loglikelihood requests:  63%|████████████████████████              | 5051/7997 [11:06<01:55, 25.47it/s][A
    Running loglikelihood requests:  63%|████████████████████████              | 5059/7997 [11:06<01:56, 25.20it/s][A
    Running loglikelihood requests:  63%|████████████████████████              | 5067/7997 [11:07<02:02, 23.94it/s][A
    Running loglikelihood requests:  63%|████████████████████████▏             | 5078/7997 [11:07<01:54, 25.45it/s][A
    Running loglikelihood requests:  64%|████████████████████████▏             | 5086/7997 [11:07<01:57, 24.78it/s][A
    Running loglikelihood requests:  64%|████████████████████████▏             | 5094/7997 [11:08<01:58, 24.46it/s][A
    Running loglikelihood requests:  64%|████████████████████████▏             | 5102/7997 [11:08<01:56, 24.77it/s][A
    Running loglikelihood requests:  64%|████████████████████████▎             | 5110/7997 [11:08<01:56, 24.84it/s][A
    Running loglikelihood requests:  64%|████████████████████████▎             | 5118/7997 [11:09<01:56, 24.67it/s][A
    Running loglikelihood requests:  64%|████████████████████████▎             | 5126/7997 [11:09<01:56, 24.67it/s][A
    Running loglikelihood requests:  64%|████████████████████████▍             | 5134/7997 [11:09<02:08, 22.33it/s][A
    Running loglikelihood requests:  64%|████████████████████████▍             | 5142/7997 [11:10<02:04, 22.88it/s][A
    Running loglikelihood requests:  64%|████████████████████████▍             | 5150/7997 [11:10<02:01, 23.48it/s][A
    Running loglikelihood requests:  64%|████████████████████████▌             | 5158/7997 [11:10<01:57, 24.08it/s][A
    Running loglikelihood requests:  65%|████████████████████████▌             | 5168/7997 [11:11<01:57, 24.03it/s][A
    Running loglikelihood requests:  65%|████████████████████████▌             | 5181/7997 [11:11<01:41, 27.71it/s][A
    Running loglikelihood requests:  65%|████████████████████████▋             | 5190/7997 [11:11<01:39, 28.11it/s][A
    Running loglikelihood requests:  65%|████████████████████████▋             | 5200/7997 [11:12<01:37, 28.65it/s][A
    Running loglikelihood requests:  65%|████████████████████████▋             | 5208/7997 [11:12<01:40, 27.83it/s][A
    Running loglikelihood requests:  65%|████████████████████████▊             | 5218/7997 [11:12<01:36, 28.71it/s][A
    Running loglikelihood requests:  65%|████████████████████████▊             | 5227/7997 [11:13<01:37, 28.41it/s][A
    Running loglikelihood requests:  65%|████████████████████████▉             | 5235/7997 [11:13<01:40, 27.61it/s][A
    Running loglikelihood requests:  66%|████████████████████████▉             | 5243/7997 [11:13<01:42, 26.88it/s][A
    Running loglikelihood requests:  66%|████████████████████████▉             | 5252/7997 [11:14<01:42, 26.79it/s][A
    Running loglikelihood requests:  66%|█████████████████████████             | 5263/7997 [11:14<01:35, 28.58it/s][A
    Running loglikelihood requests:  66%|█████████████████████████             | 5271/7997 [11:14<01:40, 27.17it/s][A
    Running loglikelihood requests:  66%|█████████████████████████             | 5279/7997 [11:15<01:43, 26.39it/s][A
    Running loglikelihood requests:  66%|█████████████████████████             | 5287/7997 [11:15<01:46, 25.49it/s][A
    Running loglikelihood requests:  66%|█████████████████████████▏            | 5295/7997 [11:15<01:46, 25.29it/s][A
    Running loglikelihood requests:  66%|█████████████████████████▏            | 5303/7997 [11:22<12:02,  3.73it/s][A
    Running loglikelihood requests:  66%|█████████████████████████▏            | 5311/7997 [11:23<09:47,  4.57it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▎            | 5319/7997 [11:23<07:24,  6.03it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▎            | 5327/7997 [11:23<05:44,  7.75it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▎            | 5336/7997 [11:24<04:21, 10.16it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▍            | 5345/7997 [11:24<03:27, 12.77it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▍            | 5353/7997 [11:24<03:00, 14.67it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▍            | 5363/7997 [11:25<02:32, 17.32it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▌            | 5371/7997 [11:25<02:36, 16.77it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▌            | 5379/7997 [11:26<02:26, 17.85it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▌            | 5387/7997 [11:26<02:13, 19.60it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▋            | 5395/7997 [11:26<02:04, 20.94it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▋            | 5405/7997 [11:27<01:53, 22.82it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▋            | 5417/7997 [11:27<01:36, 26.86it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▊            | 5425/7997 [11:27<01:36, 26.70it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▊            | 5433/7997 [11:28<01:39, 25.84it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▊            | 5444/7997 [11:28<01:28, 28.81it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▉            | 5455/7997 [11:28<01:22, 30.80it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▉            | 5463/7997 [11:28<01:27, 29.11it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▉            | 5471/7997 [11:29<01:33, 27.09it/s][A
    Running loglikelihood requests:  69%|██████████████████████████            | 5479/7997 [11:29<01:32, 27.15it/s][A
    Running loglikelihood requests:  69%|██████████████████████████            | 5487/7997 [11:29<01:34, 26.53it/s][A
    Running loglikelihood requests:  69%|██████████████████████████            | 5495/7997 [11:30<01:36, 25.99it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▏           | 5503/7997 [11:30<01:36, 25.82it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▏           | 5511/7997 [11:30<01:35, 26.16it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▏           | 5519/7997 [11:31<01:36, 25.71it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▎           | 5529/7997 [11:31<01:28, 27.90it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▎           | 5537/7997 [11:31<01:29, 27.42it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▎           | 5545/7997 [11:32<01:33, 26.33it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▍           | 5556/7997 [11:32<01:22, 29.61it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▍           | 5564/7997 [11:32<01:24, 28.65it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▍           | 5572/7997 [11:33<01:37, 24.92it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▌           | 5580/7997 [11:33<01:35, 25.39it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▌           | 5588/7997 [11:33<01:34, 25.37it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▌           | 5596/7997 [11:34<01:34, 25.40it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▋           | 5604/7997 [11:34<01:37, 24.49it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▋           | 5612/7997 [11:34<01:33, 25.48it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▋           | 5620/7997 [11:35<01:33, 25.34it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▋           | 5628/7997 [11:35<01:33, 25.46it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▊           | 5636/7997 [11:35<01:32, 25.53it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▊           | 5644/7997 [11:35<01:31, 25.69it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▊           | 5652/7997 [11:36<01:33, 25.14it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▉           | 5660/7997 [11:36<01:34, 24.77it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▉           | 5668/7997 [11:36<01:32, 25.06it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▉           | 5676/7997 [11:37<01:39, 23.38it/s][A
    Running loglikelihood requests:  71%|███████████████████████████           | 5684/7997 [11:37<01:39, 23.16it/s][A
    Running loglikelihood requests:  71%|███████████████████████████           | 5695/7997 [11:37<01:25, 26.84it/s][A
    Running loglikelihood requests:  71%|███████████████████████████           | 5705/7997 [11:38<01:21, 28.25it/s][A
    Running loglikelihood requests:  71%|███████████████████████████▏          | 5713/7997 [11:38<01:22, 27.68it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▏          | 5722/7997 [11:39<01:38, 23.01it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▏          | 5730/7997 [11:39<01:43, 21.83it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▎          | 5738/7997 [11:39<01:45, 21.34it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▎          | 5749/7997 [11:40<01:31, 24.55it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▎          | 5757/7997 [11:40<01:37, 23.00it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▍          | 5765/7997 [11:41<01:40, 22.31it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▍          | 5773/7997 [11:41<01:48, 20.50it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▍          | 5781/7997 [11:41<01:48, 20.36it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▌          | 5789/7997 [11:42<01:45, 20.95it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▌          | 5799/7997 [11:42<01:35, 22.91it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▌          | 5807/7997 [11:43<01:35, 22.94it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▋          | 5818/7997 [11:43<01:29, 24.43it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▋          | 5826/7997 [11:43<01:29, 24.33it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▋          | 5834/7997 [11:44<01:30, 23.92it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▊          | 5842/7997 [11:44<01:29, 23.96it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▊          | 5850/7997 [11:44<01:30, 23.70it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▊          | 5858/7997 [11:45<01:27, 24.55it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▊          | 5866/7997 [11:46<02:22, 14.98it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▉          | 5874/7997 [11:46<01:53, 18.78it/s][A
    Running loglikelihood requests:  74%|███████████████████████████▉          | 5882/7997 [11:46<01:35, 22.18it/s][A
    Running loglikelihood requests:  74%|███████████████████████████▉          | 5890/7997 [11:46<01:36, 21.92it/s][A
    Running loglikelihood requests:  74%|████████████████████████████          | 5900/7997 [11:47<01:30, 23.26it/s][A
    Running loglikelihood requests:  74%|████████████████████████████          | 5908/7997 [11:47<01:27, 23.89it/s][A
    Running loglikelihood requests:  74%|████████████████████████████          | 5916/7997 [11:47<01:25, 24.45it/s][A
    Running loglikelihood requests:  74%|████████████████████████████▏         | 5924/7997 [11:48<01:25, 24.25it/s][A
    Running loglikelihood requests:  74%|████████████████████████████▏         | 5935/7997 [11:48<01:14, 27.75it/s][A
    Running loglikelihood requests:  74%|████████████████████████████▏         | 5943/7997 [11:48<01:23, 24.60it/s][A
    Running loglikelihood requests:  74%|████████████████████████████▎         | 5951/7997 [11:49<01:20, 25.41it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▎         | 5961/7997 [11:49<01:15, 27.03it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▎         | 5969/7997 [11:49<01:19, 25.52it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▍         | 5977/7997 [11:50<01:23, 24.26it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▍         | 5985/7997 [11:50<01:23, 24.12it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▍         | 5993/7997 [11:50<01:21, 24.67it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▌         | 6002/7997 [11:51<01:17, 25.81it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▌         | 6010/7997 [11:51<01:17, 25.66it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▌         | 6018/7997 [11:51<01:17, 25.69it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▋         | 6029/7997 [11:52<01:10, 27.74it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▋         | 6040/7997 [11:52<01:05, 30.09it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▋         | 6048/7997 [11:52<01:07, 28.75it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▊         | 6056/7997 [11:53<01:09, 27.80it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▊         | 6064/7997 [11:53<01:11, 26.92it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▊         | 6072/7997 [11:53<01:12, 26.64it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▉         | 6080/7997 [12:04<13:16,  2.41it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▉         | 6088/7997 [12:05<10:13,  3.11it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▉         | 6096/7997 [12:05<07:33,  4.19it/s][A
    Running loglikelihood requests:  76%|█████████████████████████████         | 6104/7997 [12:05<05:40,  5.56it/s][A
    Running loglikelihood requests:  76%|█████████████████████████████         | 6112/7997 [12:06<04:18,  7.29it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████         | 6123/7997 [12:06<03:13,  9.68it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████         | 6127/7997 [12:06<02:51, 10.88it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▏        | 6134/7997 [12:07<02:37, 11.85it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▏        | 6142/7997 [12:07<02:11, 14.12it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▏        | 6150/7997 [12:07<01:52, 16.40it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▎        | 6158/7997 [12:08<01:39, 18.43it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▎        | 6168/7997 [12:08<01:28, 20.66it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▎        | 6176/7997 [12:08<01:26, 21.14it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▍        | 6186/7997 [12:09<01:16, 23.80it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▍        | 6194/7997 [12:09<01:14, 24.06it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▍        | 6202/7997 [12:09<01:17, 23.21it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▌        | 6214/7997 [12:10<01:07, 26.24it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▌        | 6222/7997 [12:10<01:05, 27.06it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▌        | 6230/7997 [12:10<01:06, 26.40it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▋        | 6238/7997 [12:11<01:05, 26.81it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▋        | 6248/7997 [12:11<01:01, 28.22it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▋        | 6256/7997 [12:11<01:02, 27.84it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▊        | 6264/7997 [12:12<01:07, 25.86it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▊        | 6272/7997 [12:12<01:14, 23.17it/s][A
    Running loglikelihood requests:  79%|█████████████████████████████▊        | 6280/7997 [12:16<05:01,  5.69it/s][A
    Running loglikelihood requests:  79%|█████████████████████████████▉        | 6288/7997 [12:18<05:12,  5.47it/s][A
    Running loglikelihood requests:  79%|█████████████████████████████▉        | 6296/7997 [12:18<03:50,  7.39it/s][A
    Running loglikelihood requests:  79%|█████████████████████████████▉        | 6304/7997 [12:18<02:50,  9.93it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████        | 6315/7997 [12:18<01:57, 14.37it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████        | 6324/7997 [12:19<01:38, 16.97it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████        | 6333/7997 [12:19<01:39, 16.77it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████▏       | 6343/7997 [12:19<01:12, 22.91it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████▏       | 6349/7997 [12:20<01:28, 18.67it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████▏       | 6355/7997 [12:20<01:27, 18.80it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▏       | 6359/7997 [12:20<01:20, 20.39it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▎       | 6368/7997 [12:20<01:11, 22.67it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▎       | 6379/7997 [12:21<00:49, 32.86it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▎       | 6385/7997 [12:21<00:57, 27.89it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▎       | 6390/7997 [12:21<01:16, 21.01it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▍       | 6394/7997 [12:21<01:10, 22.62it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▍       | 6398/7997 [12:22<01:22, 19.40it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▍       | 6406/7997 [12:22<01:14, 21.25it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▍       | 6414/7997 [12:22<01:07, 23.33it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▌       | 6425/7997 [12:23<00:58, 26.70it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▌       | 6434/7997 [12:23<00:58, 26.58it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▌       | 6444/7997 [12:23<00:53, 29.13it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▋       | 6452/7997 [12:24<00:53, 28.83it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▋       | 6460/7997 [12:24<00:57, 26.51it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▋       | 6468/7997 [12:24<00:59, 25.72it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▊       | 6476/7997 [12:30<06:20,  3.99it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▊       | 6484/7997 [12:31<05:16,  4.78it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▊       | 6492/7997 [12:31<03:53,  6.44it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▉       | 6502/7997 [12:32<02:51,  8.70it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▉       | 6511/7997 [12:32<02:16, 10.86it/s][A
    Running loglikelihood requests:  82%|██████████████████████████████▉       | 6519/7997 [12:33<01:59, 12.39it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████       | 6527/7997 [12:33<01:39, 14.79it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████       | 6535/7997 [12:33<01:30, 16.19it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████       | 6543/7997 [12:34<01:20, 18.11it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▏      | 6551/7997 [12:34<01:14, 19.42it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▏      | 6562/7997 [12:34<01:05, 21.85it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▏      | 6570/7997 [12:35<01:05, 21.80it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▎      | 6581/7997 [12:35<00:57, 24.54it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▎      | 6589/7997 [12:35<00:58, 24.01it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▎      | 6597/7997 [12:36<00:57, 24.33it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▍      | 6605/7997 [12:36<00:54, 25.52it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▍      | 6613/7997 [12:36<00:53, 26.04it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▍      | 6625/7997 [12:37<00:47, 29.09it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▌      | 6633/7997 [12:37<00:46, 29.36it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▌      | 6641/7997 [12:37<00:48, 27.91it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▌      | 6649/7997 [12:38<00:48, 27.89it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▋      | 6658/7997 [12:38<00:50, 26.67it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▋      | 6666/7997 [12:38<01:00, 21.83it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▋      | 6674/7997 [12:39<01:02, 21.03it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▊      | 6682/7997 [12:39<01:06, 19.86it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▊      | 6690/7997 [12:46<06:08,  3.55it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▊      | 6698/7997 [12:47<04:50,  4.46it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▉      | 6713/7997 [12:47<02:49,  7.58it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▉      | 6724/7997 [12:47<02:06, 10.07it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▉      | 6732/7997 [12:48<01:44, 12.06it/s][A
    Running loglikelihood requests:  84%|████████████████████████████████      | 6743/7997 [12:48<01:21, 15.37it/s][A
    Running loglikelihood requests:  84%|████████████████████████████████      | 6753/7997 [12:48<01:07, 18.30it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▏     | 6762/7997 [12:48<00:58, 21.01it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▏     | 6772/7997 [12:49<00:52, 23.47it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▏     | 6782/7997 [12:49<00:46, 26.08it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▎     | 6790/7997 [12:49<00:46, 25.85it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▎     | 6798/7997 [12:50<00:46, 26.05it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▎     | 6808/7997 [12:50<00:44, 26.53it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▍     | 6816/7997 [12:50<00:42, 28.00it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▍     | 6825/7997 [12:50<00:40, 28.84it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▍     | 6834/7997 [12:51<00:40, 28.88it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▌     | 6845/7997 [12:51<00:36, 31.99it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▌     | 6856/7997 [12:51<00:33, 33.88it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▌     | 6864/7997 [12:52<00:35, 32.20it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▋     | 6872/7997 [12:52<00:35, 31.93it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▋     | 6880/7997 [12:54<02:02,  9.09it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▋     | 6890/7997 [13:00<04:42,  3.93it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▊     | 6898/7997 [13:01<03:53,  4.72it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▊     | 6906/7997 [13:01<02:54,  6.24it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▊     | 6914/7997 [13:01<02:15,  7.97it/s][A
    Running loglikelihood requests:  87%|████████████████████████████████▉     | 6925/7997 [13:02<01:35, 11.25it/s][A
    Running loglikelihood requests:  87%|████████████████████████████████▉     | 6937/7997 [13:02<01:08, 15.39it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████     | 6947/7997 [13:02<00:56, 18.57it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████     | 6958/7997 [13:02<00:45, 22.62it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████     | 6970/7997 [13:03<00:40, 25.41it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████▏    | 6978/7997 [13:03<00:41, 24.74it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████▏    | 6986/7997 [13:03<00:42, 24.00it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████▏    | 6995/7997 [13:04<00:39, 25.56it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▎    | 7008/7997 [13:04<00:32, 30.34it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▎    | 7016/7997 [13:04<00:32, 30.59it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▍    | 7024/7997 [13:05<00:31, 30.41it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▍    | 7032/7997 [13:05<00:33, 29.14it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▍    | 7040/7997 [13:07<01:33, 10.19it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▍    | 7048/7997 [13:07<01:16, 12.47it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▌    | 7056/7997 [13:07<01:03, 14.78it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▌    | 7070/7997 [13:08<00:43, 21.41it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▋    | 7078/7997 [13:08<00:40, 22.93it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▋    | 7088/7997 [13:08<00:35, 25.94it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▋    | 7098/7997 [13:09<00:31, 28.76it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▊    | 7109/7997 [13:09<00:27, 32.20it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▊    | 7120/7997 [13:09<00:26, 33.64it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▉    | 7130/7997 [13:09<00:25, 34.29it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▉    | 7141/7997 [13:10<00:23, 36.22it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▉    | 7150/7997 [13:10<00:25, 33.49it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████    | 7160/7997 [13:10<00:25, 32.39it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████    | 7173/7997 [13:11<00:23, 35.67it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████    | 7181/7997 [13:11<00:27, 30.13it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▏   | 7189/7997 [13:11<00:27, 29.65it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▏   | 7197/7997 [13:12<00:38, 20.90it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▏   | 7205/7997 [13:12<00:34, 23.06it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▎   | 7214/7997 [13:13<00:33, 23.54it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▎   | 7222/7997 [13:13<00:31, 24.26it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▎   | 7231/7997 [13:13<00:30, 25.23it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▍   | 7239/7997 [13:14<00:31, 24.28it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▍   | 7249/7997 [13:14<00:28, 26.59it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▍   | 7259/7997 [13:14<00:27, 26.86it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▌   | 7270/7997 [13:15<00:25, 28.21it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▌   | 7281/7997 [13:15<00:24, 29.10it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▋   | 7289/7997 [13:15<00:25, 27.34it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▋   | 7302/7997 [13:16<00:22, 31.45it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▋   | 7313/7997 [13:16<00:20, 33.72it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▊   | 7321/7997 [13:16<00:20, 32.49it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▊   | 7329/7997 [13:16<00:21, 31.13it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▊   | 7337/7997 [13:24<03:11,  3.44it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▉   | 7345/7997 [13:25<02:40,  4.06it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▉   | 7356/7997 [13:26<01:45,  6.08it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▉   | 7364/7997 [13:26<01:21,  7.80it/s][A
    Running loglikelihood requests:  92%|███████████████████████████████████   | 7373/7997 [13:26<01:04,  9.73it/s][A
    Running loglikelihood requests:  92%|███████████████████████████████████   | 7385/7997 [13:27<00:46, 13.22it/s][A
    Running loglikelihood requests:  92%|███████████████████████████████████▏  | 7395/7997 [13:27<00:36, 16.45it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▏  | 7406/7997 [13:27<00:29, 20.26it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▏  | 7414/7997 [13:27<00:28, 20.24it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▎  | 7426/7997 [13:28<00:22, 25.73it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▎  | 7437/7997 [13:28<00:19, 28.75it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▍  | 7450/7997 [13:28<00:16, 33.04it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▍  | 7460/7997 [13:29<00:15, 34.89it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▍  | 7469/7997 [13:29<00:15, 34.02it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▌  | 7478/7997 [13:29<00:15, 34.54it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▌  | 7486/7997 [13:29<00:15, 32.08it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▌  | 7497/7997 [13:30<00:14, 33.54it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▋  | 7505/7997 [13:33<01:06,  7.35it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▋  | 7519/7997 [13:33<00:42, 11.13it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▊  | 7532/7997 [13:34<00:32, 14.26it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▊  | 7540/7997 [13:34<00:28, 16.19it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▉  | 7554/7997 [13:34<00:20, 22.08it/s][A
    Running loglikelihood requests:  95%|███████████████████████████████████▉  | 7566/7997 [13:35<00:16, 26.51it/s][A
    Running loglikelihood requests:  95%|███████████████████████████████████▉  | 7574/7997 [13:35<00:15, 27.98it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████  | 7584/7997 [13:35<00:13, 30.01it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████  | 7594/7997 [13:35<00:13, 30.35it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████▏ | 7608/7997 [13:36<00:10, 37.85it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████▏ | 7616/7997 [13:36<00:10, 36.04it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████▏ | 7624/7997 [13:44<01:42,  3.62it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████▎ | 7633/7997 [13:45<01:19,  4.56it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▎ | 7643/7997 [13:45<00:55,  6.35it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▎ | 7651/7997 [13:45<00:43,  7.99it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▍ | 7662/7997 [13:46<00:30, 11.13it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▍ | 7672/7997 [13:46<00:22, 14.33it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▍ | 7680/7997 [13:46<00:18, 17.05it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▌ | 7700/7997 [13:46<00:10, 27.55it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▋ | 7713/7997 [13:47<00:08, 32.44it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▋ | 7722/7997 [13:47<00:08, 33.84it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▋ | 7733/7997 [13:47<00:09, 26.75it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▊ | 7744/7997 [13:48<00:08, 30.99it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▊ | 7755/7997 [13:48<00:06, 35.00it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▉ | 7770/7997 [13:48<00:05, 41.89it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▉ | 7785/7997 [13:48<00:04, 47.79it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████ | 7802/7997 [13:49<00:03, 49.48it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████ | 7812/7997 [13:49<00:03, 50.34it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▏| 7820/7997 [13:49<00:03, 46.12it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▏| 7828/7997 [13:49<00:03, 42.41it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▏| 7837/7997 [13:50<00:05, 31.09it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▎| 7852/7997 [13:50<00:03, 38.39it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▍| 7870/7997 [13:50<00:02, 49.59it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▍| 7888/7997 [13:50<00:01, 58.18it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▌| 7905/7997 [13:51<00:01, 60.54it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▌| 7918/7997 [13:51<00:01, 60.47it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▋| 7926/7997 [13:51<00:01, 53.18it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▋| 7937/7997 [13:51<00:01, 44.99it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▊| 7949/7997 [13:52<00:01, 46.48it/s][A
    Running loglikelihood requests: 100%|█████████████████████████████████████▊| 7967/7997 [13:52<00:00, 59.49it/s][A
    Running loglikelihood requests: 100%|█████████████████████████████████████▉| 7976/7997 [13:52<00:00, 36.11it/s][A
    Running loglikelihood requests: 100%|██████████████████████████████████████| 7997/7997 [13:53<00:00,  9.59it/s][A
    Python(70454) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    fatal: not a git repository (or any of the parent directories): .git
    Python(70455) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(70456) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(70457) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(70458) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(70462) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(70463) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(70464) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(70465) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(70466) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(70467) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.


    
    === bf16 ===
    arc_easy {'acc,none': 0.607, 'acc_stderr,none': 0.015, 'acc_norm,none': 0.57, 'acc_norm_stderr,none': 0.016}
    hellaswag {'acc,none': 0.382, 'acc_stderr,none': 0.015, 'acc_norm,none': 0.475, 'acc_norm_stderr,none': 0.016}


    /Users/ayush.ganatra/model_inference/myenv/lib/python3.14/site-packages/transformers/quantizers/auto.py:279: UserWarning: You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading already has a `quantization_config` attribute. The `quantization_config` from the model will be used.However, loading attributes (e.g. ['dequantize', 'use_optimized_inference']) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored.
      warnings.warn(warning_msg)
    Applying quantization config: 100%|████████████████████████████████████████| 196/196 [00:00<00:00, 8751.53it/s]
    
    Compressing model: 100%|██████████████████████████████████████████████████| 196/196 [00:00<00:00, 17865.56it/s][A



    Loading weights:   0%|          | 0/702 [00:00<?, ?it/s]


    
      0%|                                                                                 | 0/1000 [00:00<?, ?it/s][A
      9%|██████▍                                                                | 90/1000 [00:00<00:01, 895.67it/s][A
     19%|█████████████                                                         | 186/1000 [00:00<00:00, 929.94it/s][A
     28%|███████████████████▋                                                  | 281/1000 [00:00<00:00, 937.20it/s][A
     38%|██████████████████████████▍                                           | 377/1000 [00:00<00:00, 945.87it/s][A
     47%|█████████████████████████████████▏                                    | 474/1000 [00:00<00:00, 953.26it/s][A
     57%|███████████████████████████████████████▉                              | 570/1000 [00:00<00:00, 937.42it/s][A
     66%|██████████████████████████████████████████████▍                       | 664/1000 [00:00<00:00, 932.87it/s][A
     76%|█████████████████████████████████████████████████████                 | 758/1000 [00:00<00:00, 927.71it/s][A
     85%|███████████████████████████████████████████████████████████▌          | 851/1000 [00:00<00:00, 923.09it/s][A
    100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 929.97it/s][A
    
      0%|                                                                                 | 0/1000 [00:00<?, ?it/s][A
     25%|█████████████████▍                                                   | 253/1000 [00:00<00:00, 2522.46it/s][A
     52%|███████████████████████████████████▌                                 | 515/1000 [00:00<00:00, 2576.98it/s][A
     77%|█████████████████████████████████████████████████████▎               | 773/1000 [00:00<00:00, 1321.97it/s][A
    100%|████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 1417.04it/s][A
    
    Tokenizing inputs:   0%|                                                              | 0/7997 [00:00<?, ?it/s][A
    Tokenizing inputs:   6%|██▊                                               | 454/7997 [00:00<00:01, 4538.73it/s][A
    Tokenizing inputs:  13%|██████▎                                          | 1031/7997 [00:00<00:01, 5262.30it/s][A
    Tokenizing inputs:  19%|█████████▌                                       | 1558/7997 [00:00<00:01, 5154.20it/s][A
    Tokenizing inputs:  26%|████████████▋                                    | 2074/7997 [00:00<00:01, 4525.77it/s][A
    Tokenizing inputs:  34%|████████████████▊                                | 2742/7997 [00:00<00:01, 5232.35it/s][A
    Tokenizing inputs:  43%|█████████████████████▏                           | 3456/7997 [00:00<00:00, 5839.86it/s][A
    Tokenizing inputs:  51%|█████████████████████████                        | 4092/7997 [00:00<00:00, 6002.15it/s][A
    Tokenizing inputs:  59%|████████████████████████████▉                    | 4729/7997 [00:00<00:00, 6112.62it/s][A
    Tokenizing inputs:  67%|████████████████████████████████▉                | 5369/7997 [00:00<00:00, 6200.48it/s][A
    Tokenizing inputs:  75%|████████████████████████████████████▉            | 6032/7997 [00:01<00:00, 6330.38it/s][A
    Tokenizing inputs:  84%|█████████████████████████████████████████        | 6692/7997 [00:01<00:00, 6410.32it/s][A
    Tokenizing inputs:  92%|████████████████████████████████████████████▉    | 7336/7997 [00:01<00:00, 6406.05it/s][A
    Tokenizing inputs: 100%|█████████████████████████████████████████████████| 7997/7997 [00:01<00:00, 5890.52it/s][A
    
    Dispatching model: 100%|██████████████████████████████████████████████████| 427/427 [00:00<00:00, 23103.90it/s][A
    
    
    Decompressing model:   0%|                                                             | 0/196 [00:00<?, ?it/s][A[A
    
    Decompressing model:   1%|▎                                                    | 1/196 [00:02<08:24,  2.59s/it][A[A
    
    Decompressing model:   1%|▌                                                    | 2/196 [00:06<11:49,  3.66s/it][A[A
    
    Decompressing model:   3%|█▎                                                   | 5/196 [00:07<03:45,  1.18s/it][A[A
    
    Decompressing model:   6%|██▉                                                 | 11/196 [00:07<01:13,  2.50it/s][A[A
    
    Decompressing model:  11%|█████▌                                              | 21/196 [00:07<00:27,  6.25it/s][A[A
    
    Decompressing model:  17%|████████▊                                           | 33/196 [00:07<00:13, 12.15it/s][A[A
    
    Decompressing model:  25%|█████████████                                       | 49/196 [00:08<00:06, 22.33it/s][A[A
    
    Decompressing model:  34%|█████████████████▌                                  | 66/196 [00:08<00:03, 35.50it/s][A[A
    
    Decompressing model:  40%|████████████████████▉                               | 79/196 [00:08<00:04, 26.96it/s][A[A
    
    Decompressing model:  49%|█████████████████████████▋                          | 97/196 [00:08<00:02, 39.96it/s][A[A
    
    Decompressing model:  60%|██████████████████████████████▋                    | 118/196 [00:09<00:01, 58.43it/s][A[A
    
    Decompressing model:  70%|███████████████████████████████████▉               | 138/196 [00:09<00:00, 77.15it/s][A[A
    
    Decompressing model:  80%|████████████████████████████████████████▊          | 157/196 [00:09<00:00, 94.88it/s][A[A
    
    Decompressing model:  89%|█████████████████████████████████████████████▎     | 174/196 [00:13<00:01, 13.50it/s][A[A
    
    Decompressing model: 100%|███████████████████████████████████████████████████| 196/196 [00:22<00:00,  8.62it/s][A[A
    
    Running loglikelihood requests:   0%|                                     | 1/7997 [01:17<172:25:29, 77.63s/it][A
    Running loglikelihood requests:   0%|                                      | 9/7997 [01:36<18:33:20,  8.36s/it][A
    Running loglikelihood requests:   0%|                                      | 15/7997 [01:36<9:25:51,  4.25s/it][A
    Running loglikelihood requests:   0%|                                      | 17/7997 [01:44<9:23:32,  4.24s/it][A
    Running loglikelihood requests:   0%|                                      | 25/7997 [01:52<5:34:49,  2.52s/it][A
    Running loglikelihood requests:   1%|▏                                     | 40/7997 [01:57<2:43:16,  1.23s/it][A
    Running loglikelihood requests:   1%|▏                                     | 47/7997 [01:57<1:57:59,  1.12it/s][A
    Running loglikelihood requests:   1%|▏                                     | 50/7997 [01:59<1:53:38,  1.17it/s][A
    Running loglikelihood requests:   1%|▎                                     | 56/7997 [02:01<1:34:25,  1.40it/s][A
    Running loglikelihood requests:   1%|▎                                     | 58/7997 [02:01<1:24:03,  1.57it/s][A
    Running loglikelihood requests:   1%|▎                                     | 64/7997 [02:04<1:17:34,  1.70it/s][A
    Running loglikelihood requests:   1%|▎                                     | 72/7997 [02:14<1:54:09,  1.16it/s][A
    Running loglikelihood requests:   1%|▍                                     | 80/7997 [02:16<1:25:11,  1.55it/s][A
    Running loglikelihood requests:   1%|▍                                     | 88/7997 [02:29<2:09:30,  1.02it/s][A
    Running loglikelihood requests:   1%|▍                                     | 91/7997 [02:30<1:50:28,  1.19it/s][A
    Running loglikelihood requests:   1%|▍                                     | 93/7997 [02:30<1:37:37,  1.35it/s][A
    Running loglikelihood requests:   1%|▍                                     | 96/7997 [02:36<2:13:31,  1.01s/it][A
    Running loglikelihood requests:   1%|▍                                    | 107/7997 [02:37<1:11:30,  1.84it/s][A
    Running loglikelihood requests:   1%|▌                                      | 115/7997 [02:38<51:03,  2.57it/s][A
    Running loglikelihood requests:   2%|▌                                      | 123/7997 [02:39<37:45,  3.48it/s][A
    Running loglikelihood requests:   2%|▋                                      | 131/7997 [02:40<29:33,  4.43it/s][A
    Running loglikelihood requests:   2%|▋                                      | 139/7997 [02:40<24:26,  5.36it/s][A
    Running loglikelihood requests:   2%|▋                                      | 147/7997 [02:41<20:40,  6.33it/s][A
    Running loglikelihood requests:   2%|▊                                      | 155/7997 [02:42<17:50,  7.33it/s][A
    Running loglikelihood requests:   2%|▊                                      | 163/7997 [02:43<16:05,  8.12it/s][A
    Running loglikelihood requests:   2%|▊                                      | 171/7997 [02:43<14:51,  8.78it/s][A
    Running loglikelihood requests:   2%|▊                                      | 179/7997 [02:44<13:40,  9.53it/s][A
    Running loglikelihood requests:   2%|▉                                      | 187/7997 [02:45<12:44, 10.22it/s][A
    Running loglikelihood requests:   2%|▉                                      | 195/7997 [02:45<12:15, 10.61it/s][A
    Running loglikelihood requests:   3%|▉                                      | 203/7997 [02:46<11:52, 10.94it/s][A
    Running loglikelihood requests:   3%|█                                      | 213/7997 [02:47<10:39, 12.17it/s][A
    Running loglikelihood requests:   3%|█                                      | 221/7997 [02:47<10:45, 12.05it/s][A
    Running loglikelihood requests:   3%|█                                      | 229/7997 [02:48<10:35, 12.23it/s][A
    Running loglikelihood requests:   3%|█▏                                     | 237/7997 [02:49<10:33, 12.26it/s][A
    Running loglikelihood requests:   3%|█▏                                     | 245/7997 [02:49<10:43, 12.05it/s][A
    Running loglikelihood requests:   3%|█▏                                     | 253/7997 [02:50<10:38, 12.13it/s][A
    Running loglikelihood requests:   3%|█▎                                     | 261/7997 [02:51<12:09, 10.61it/s][A
    Running loglikelihood requests:   3%|█▎                                     | 269/7997 [02:52<11:53, 10.83it/s][A
    Running loglikelihood requests:   3%|█▎                                     | 277/7997 [02:52<12:01, 10.70it/s][A
    Running loglikelihood requests:   4%|█▍                                     | 288/7997 [02:53<10:17, 12.49it/s][A
    Running loglikelihood requests:   4%|█▍                                     | 296/7997 [02:54<10:17, 12.47it/s][A
    Running loglikelihood requests:   4%|█▍                                     | 304/7997 [02:57<24:13,  5.29it/s][A
    Running loglikelihood requests:   4%|█▌                                     | 312/7997 [03:01<32:25,  3.95it/s][A
    Running loglikelihood requests:   4%|█▌                                     | 320/7997 [03:02<30:19,  4.22it/s][A
    Running loglikelihood requests:   4%|█▌                                     | 328/7997 [03:04<27:41,  4.62it/s][A
    Running loglikelihood requests:   4%|█▋                                     | 336/7997 [03:05<24:14,  5.27it/s][A
    Running loglikelihood requests:   4%|█▋                                     | 344/7997 [03:05<21:08,  6.03it/s][A
    Running loglikelihood requests:   4%|█▋                                     | 352/7997 [03:06<18:40,  6.82it/s][A
    Running loglikelihood requests:   5%|█▊                                     | 360/7997 [03:07<16:49,  7.56it/s][A
    Running loglikelihood requests:   5%|█▊                                     | 368/7997 [03:08<14:36,  8.71it/s][A
    Running loglikelihood requests:   5%|█▊                                     | 376/7997 [03:08<13:06,  9.68it/s][A
    Running loglikelihood requests:   5%|█▊                                     | 384/7997 [03:09<13:19,  9.53it/s][A
    Running loglikelihood requests:   5%|█▉                                     | 392/7997 [03:10<12:20, 10.27it/s][A
    Running loglikelihood requests:   5%|█▉                                     | 400/7997 [03:10<11:37, 10.89it/s][A
    Running loglikelihood requests:   5%|█▉                                     | 409/7997 [03:11<10:43, 11.79it/s][A
    Running loglikelihood requests:   5%|██                                     | 417/7997 [03:12<10:20, 12.22it/s][A
    Running loglikelihood requests:   5%|██                                     | 425/7997 [03:12<10:07, 12.46it/s][A
    Running loglikelihood requests:   5%|██                                     | 433/7997 [03:13<09:45, 12.92it/s][A
    Running loglikelihood requests:   6%|██▏                                    | 441/7997 [03:13<09:34, 13.15it/s][A
    Running loglikelihood requests:   6%|██▏                                    | 449/7997 [03:14<09:33, 13.17it/s][A
    Running loglikelihood requests:   6%|██▏                                    | 457/7997 [03:15<09:34, 13.11it/s][A
    Running loglikelihood requests:   6%|██▎                                    | 465/7997 [03:15<09:33, 13.14it/s][A
    Running loglikelihood requests:   6%|██▎                                    | 473/7997 [03:16<09:34, 13.10it/s][A
    Running loglikelihood requests:   6%|██▎                                    | 481/7997 [03:17<09:54, 12.64it/s][A
    Running loglikelihood requests:   6%|██▍                                    | 489/7997 [03:17<09:59, 12.53it/s][A
    Running loglikelihood requests:   6%|██▍                                    | 497/7997 [03:18<10:37, 11.76it/s][A
    Running loglikelihood requests:   6%|██▍                                    | 505/7997 [03:19<10:55, 11.42it/s][A
    Running loglikelihood requests:   6%|██▌                                    | 513/7997 [03:20<12:23, 10.06it/s][A
    Running loglikelihood requests:   7%|██▌                                    | 521/7997 [03:21<12:30,  9.96it/s][A
    Running loglikelihood requests:   7%|██▌                                    | 532/7997 [03:21<11:19, 10.99it/s][A
    Running loglikelihood requests:   7%|██▋                                    | 544/7997 [03:22<09:26, 13.16it/s][A
    Running loglikelihood requests:   7%|██▋                                    | 552/7997 [03:23<09:50, 12.61it/s][A
    Running loglikelihood requests:   7%|██▋                                    | 560/7997 [03:23<10:03, 12.31it/s][A
    Running loglikelihood requests:   7%|██▊                                    | 568/7997 [03:24<10:59, 11.26it/s][A
    Running loglikelihood requests:   7%|██▊                                    | 576/7997 [03:25<10:35, 11.67it/s][A
    Running loglikelihood requests:   7%|██▊                                    | 585/7997 [03:25<09:43, 12.71it/s][A
    Running loglikelihood requests:   7%|██▉                                    | 595/7997 [03:26<09:12, 13.40it/s][A
    Running loglikelihood requests:   8%|██▉                                    | 603/7997 [03:27<09:16, 13.29it/s][A
    Running loglikelihood requests:   8%|██▉                                    | 611/7997 [03:27<09:10, 13.42it/s][A
    Running loglikelihood requests:   8%|███                                    | 619/7997 [03:28<09:31, 12.92it/s][A
    Running loglikelihood requests:   8%|███                                    | 627/7997 [03:29<09:31, 12.89it/s][A
    Running loglikelihood requests:   8%|███                                    | 635/7997 [03:29<09:23, 13.06it/s][A
    Running loglikelihood requests:   8%|███▏                                   | 643/7997 [03:30<09:01, 13.59it/s][A
    Running loglikelihood requests:   8%|███▏                                   | 651/7997 [03:30<08:52, 13.79it/s][A
    Running loglikelihood requests:   8%|███▏                                   | 659/7997 [03:31<08:50, 13.83it/s][A
    Running loglikelihood requests:   8%|███▎                                   | 667/7997 [03:31<08:41, 14.05it/s][A
    Running loglikelihood requests:   8%|███▎                                   | 675/7997 [03:32<08:40, 14.06it/s][A
    Running loglikelihood requests:   9%|███▎                                   | 683/7997 [03:33<08:26, 14.44it/s][A
    Running loglikelihood requests:   9%|███▎                                   | 691/7997 [03:33<08:19, 14.62it/s][A
    Running loglikelihood requests:   9%|███▍                                   | 699/7997 [03:34<08:52, 13.70it/s][A
    Running loglikelihood requests:   9%|███▍                                   | 707/7997 [03:34<08:45, 13.88it/s][A
    Running loglikelihood requests:   9%|███▍                                   | 715/7997 [03:35<08:30, 14.26it/s][A
    Running loglikelihood requests:   9%|███▌                                   | 723/7997 [03:35<08:23, 14.44it/s][A
    Running loglikelihood requests:   9%|███▌                                   | 731/7997 [03:36<08:17, 14.62it/s][A
    Running loglikelihood requests:   9%|███▌                                   | 739/7997 [03:36<08:31, 14.20it/s][A
    Running loglikelihood requests:   9%|███▋                                   | 747/7997 [03:37<08:28, 14.25it/s][A
    Running loglikelihood requests:   9%|███▋                                   | 755/7997 [03:38<08:22, 14.42it/s][A
    Running loglikelihood requests:  10%|███▋                                   | 763/7997 [03:38<08:14, 14.64it/s][A
    Running loglikelihood requests:  10%|███▊                                   | 771/7997 [03:39<08:06, 14.84it/s][A
    Running loglikelihood requests:  10%|███▊                                   | 779/7997 [03:39<08:03, 14.93it/s][A
    Running loglikelihood requests:  10%|███▊                                   | 787/7997 [03:40<08:25, 14.27it/s][A
    Running loglikelihood requests:  10%|███▉                                   | 795/7997 [03:40<08:16, 14.52it/s][A
    Running loglikelihood requests:  10%|███▉                                   | 803/7997 [03:41<08:26, 14.19it/s][A
    Running loglikelihood requests:  10%|███▉                                   | 811/7997 [03:41<08:21, 14.33it/s][A
    Running loglikelihood requests:  10%|███▉                                   | 819/7997 [03:42<08:18, 14.41it/s][A
    Running loglikelihood requests:  10%|████                                   | 827/7997 [03:42<08:08, 14.69it/s][A
    Running loglikelihood requests:  10%|████                                   | 835/7997 [03:43<08:24, 14.19it/s][A
    Running loglikelihood requests:  11%|████                                   | 843/7997 [03:44<08:19, 14.31it/s][A
    Running loglikelihood requests:  11%|████▏                                  | 857/7997 [03:44<06:46, 17.55it/s][A
    Running loglikelihood requests:  11%|████▏                                  | 865/7997 [03:45<07:00, 16.95it/s][A
    Running loglikelihood requests:  11%|████▎                                  | 873/7997 [03:45<07:14, 16.40it/s][A
    Running loglikelihood requests:  11%|████▎                                  | 881/7997 [03:46<07:20, 16.14it/s][A
    Running loglikelihood requests:  11%|████▎                                  | 889/7997 [03:46<07:25, 15.96it/s][A
    Running loglikelihood requests:  11%|████▎                                  | 897/7997 [03:47<07:34, 15.61it/s][A
    Running loglikelihood requests:  11%|████▍                                  | 905/7997 [03:47<07:33, 15.65it/s][A
    Running loglikelihood requests:  11%|████▍                                  | 913/7997 [03:48<07:33, 15.63it/s][A
    Running loglikelihood requests:  12%|████▍                                  | 921/7997 [03:48<07:54, 14.92it/s][A
    Running loglikelihood requests:  12%|████▌                                  | 929/7997 [03:49<07:52, 14.97it/s][A
    Running loglikelihood requests:  12%|████▌                                  | 937/7997 [03:49<07:46, 15.14it/s][A
    Running loglikelihood requests:  12%|████▌                                  | 947/7997 [03:50<07:22, 15.92it/s][A
    Running loglikelihood requests:  12%|████▋                                  | 955/7997 [03:51<07:24, 15.84it/s][A
    Running loglikelihood requests:  12%|████▋                                  | 963/7997 [03:51<07:46, 15.07it/s][A
    Running loglikelihood requests:  12%|████▋                                  | 971/7997 [03:52<07:47, 15.03it/s][A
    Running loglikelihood requests:  12%|████▊                                  | 979/7997 [03:52<07:47, 15.02it/s][A
    Running loglikelihood requests:  12%|████▊                                  | 987/7997 [03:53<08:09, 14.33it/s][A
    Running loglikelihood requests:  12%|████▊                                  | 995/7997 [03:53<08:11, 14.26it/s][A
    Running loglikelihood requests:  13%|████▊                                 | 1003/7997 [03:54<08:37, 13.51it/s][A
    Running loglikelihood requests:  13%|████▊                                 | 1011/7997 [03:55<08:21, 13.93it/s][A
    Running loglikelihood requests:  13%|████▊                                 | 1019/7997 [03:55<08:11, 14.20it/s][A
    Running loglikelihood requests:  13%|████▉                                 | 1027/7997 [03:56<08:11, 14.19it/s][A
    Running loglikelihood requests:  13%|████▉                                 | 1037/7997 [03:56<08:03, 14.40it/s][A
    Running loglikelihood requests:  13%|████▉                                 | 1045/7997 [03:57<08:08, 14.24it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1053/7997 [03:57<07:57, 14.54it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1061/7997 [03:59<10:08, 11.39it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1069/7997 [04:00<12:12,  9.46it/s][A
    Running loglikelihood requests:  13%|█████                                 | 1077/7997 [04:01<12:21,  9.34it/s][A
    Running loglikelihood requests:  14%|█████▏                                | 1085/7997 [04:11<53:14,  2.16it/s][A
    Running loglikelihood requests:  14%|█████▏                                | 1093/7997 [04:14<51:01,  2.25it/s][A
    Running loglikelihood requests:  14%|█████▏                                | 1101/7997 [04:15<38:28,  2.99it/s][A
    Running loglikelihood requests:  14%|█████▎                                | 1109/7997 [04:15<28:30,  4.03it/s][A
    Running loglikelihood requests:  14%|█████▎                                | 1117/7997 [04:16<22:07,  5.18it/s][A
    Running loglikelihood requests:  14%|█████▎                                | 1128/7997 [04:16<15:46,  7.26it/s][A
    Running loglikelihood requests:  14%|█████▍                                | 1136/7997 [04:17<13:39,  8.38it/s][A
    Running loglikelihood requests:  14%|█████▍                                | 1144/7997 [04:17<12:06,  9.43it/s][A
    Running loglikelihood requests:  14%|█████▍                                | 1152/7997 [04:18<11:26,  9.97it/s][A
    Running loglikelihood requests:  15%|█████▌                                | 1160/7997 [04:19<11:36,  9.82it/s][A
    Running loglikelihood requests:  15%|█████▌                                | 1168/7997 [04:20<11:36,  9.80it/s][A
    Running loglikelihood requests:  15%|█████▌                                | 1176/7997 [04:21<11:47,  9.65it/s][A
    Running loglikelihood requests:  15%|█████▋                                | 1184/7997 [04:22<13:42,  8.28it/s][A
    Running loglikelihood requests:  15%|█████▋                                | 1192/7997 [04:23<12:55,  8.77it/s][A
    Running loglikelihood requests:  15%|█████▋                                | 1200/7997 [04:23<12:23,  9.14it/s][A
    Running loglikelihood requests:  15%|█████▋                                | 1208/7997 [04:24<11:35,  9.76it/s][A
    Running loglikelihood requests:  15%|█████▊                                | 1216/7997 [04:25<10:42, 10.55it/s][A
    Running loglikelihood requests:  15%|█████▊                                | 1226/7997 [04:25<09:31, 11.85it/s][A
    Running loglikelihood requests:  15%|█████▉                                | 1237/7997 [04:26<09:21, 12.04it/s][A
    Running loglikelihood requests:  16%|█████▉                                | 1245/7997 [04:27<09:41, 11.60it/s][A
    Running loglikelihood requests:  16%|█████▉                                | 1253/7997 [04:28<10:08, 11.08it/s][A
    Running loglikelihood requests:  16%|█████▉                                | 1261/7997 [04:29<10:10, 11.03it/s][A
    Running loglikelihood requests:  16%|██████                                | 1269/7997 [04:29<09:52, 11.36it/s][A
    Running loglikelihood requests:  16%|██████                                | 1277/7997 [04:30<09:45, 11.48it/s][A
    Running loglikelihood requests:  16%|██████                                | 1285/7997 [04:37<36:08,  3.10it/s][A
    Running loglikelihood requests:  16%|██████▏                               | 1293/7997 [04:42<47:35,  2.35it/s][A
    Running loglikelihood requests:  16%|██████▏                               | 1301/7997 [04:48<54:52,  2.03it/s][A
    Running loglikelihood requests:  16%|█████▉                              | 1309/7997 [04:54<1:03:46,  1.75it/s][A
    Running loglikelihood requests:  17%|█████▉                              | 1320/7997 [05:04<1:18:30,  1.42it/s][A
    Running loglikelihood requests:  17%|█████▉                              | 1328/7997 [05:06<1:03:58,  1.74it/s][A
    Running loglikelihood requests:  17%|██████▎                               | 1339/7997 [05:06<43:02,  2.58it/s][A
    Running loglikelihood requests:  17%|██████▍                               | 1347/7997 [05:07<33:42,  3.29it/s][A
    Running loglikelihood requests:  17%|██████▍                               | 1355/7997 [05:08<26:36,  4.16it/s][A
    Running loglikelihood requests:  17%|██████▍                               | 1363/7997 [05:08<21:02,  5.25it/s][A
    Running loglikelihood requests:  17%|██████▌                               | 1371/7997 [05:09<17:13,  6.41it/s][A
    Running loglikelihood requests:  17%|██████▌                               | 1379/7997 [05:09<14:09,  7.79it/s][A
    Running loglikelihood requests:  17%|██████▌                               | 1387/7997 [05:10<11:59,  9.18it/s][A
    Running loglikelihood requests:  17%|██████▋                               | 1395/7997 [05:10<10:53, 10.10it/s][A
    Running loglikelihood requests:  18%|██████▋                               | 1403/7997 [05:11<09:47, 11.22it/s][A
    Running loglikelihood requests:  18%|██████▋                               | 1411/7997 [05:11<09:05, 12.06it/s][A
    Running loglikelihood requests:  18%|██████▋                               | 1419/7997 [05:12<08:31, 12.86it/s][A
    Running loglikelihood requests:  18%|██████▊                               | 1427/7997 [05:12<08:25, 13.01it/s][A
    Running loglikelihood requests:  18%|██████▊                               | 1435/7997 [05:13<07:50, 13.94it/s][A
    Running loglikelihood requests:  18%|██████▊                               | 1444/7997 [05:13<07:14, 15.07it/s][A
    Running loglikelihood requests:  18%|██████▉                               | 1452/7997 [05:14<06:58, 15.64it/s][A
    Running loglikelihood requests:  18%|██████▉                               | 1460/7997 [05:14<07:05, 15.37it/s][A
    Running loglikelihood requests:  18%|██████▉                               | 1468/7997 [05:15<06:55, 15.72it/s][A
    Running loglikelihood requests:  18%|███████                               | 1476/7997 [05:15<06:47, 16.01it/s][A
    Running loglikelihood requests:  19%|███████                               | 1484/7997 [05:16<06:45, 16.07it/s][A
    Running loglikelihood requests:  19%|███████                               | 1492/7997 [05:16<07:07, 15.20it/s][A
    Running loglikelihood requests:  19%|███████▏                              | 1500/7997 [05:17<07:07, 15.20it/s][A
    Running loglikelihood requests:  19%|███████▏                              | 1508/7997 [05:23<27:33,  3.93it/s][A
    Running loglikelihood requests:  19%|███████▏                              | 1516/7997 [05:26<33:05,  3.26it/s][A
    Running loglikelihood requests:  19%|███████▏                              | 1524/7997 [05:26<25:22,  4.25it/s][A
    Running loglikelihood requests:  19%|███████▎                              | 1532/7997 [05:27<19:17,  5.58it/s][A
    Running loglikelihood requests:  19%|███████▎                              | 1540/7997 [05:27<15:42,  6.85it/s][A
    Running loglikelihood requests:  19%|███████▎                              | 1548/7997 [05:28<13:19,  8.07it/s][A
    Running loglikelihood requests:  19%|███████▍                              | 1556/7997 [05:28<11:18,  9.49it/s][A
    Running loglikelihood requests:  20%|███████▍                              | 1567/7997 [05:29<09:22, 11.43it/s][A
    Running loglikelihood requests:  20%|███████▍                              | 1575/7997 [05:30<08:33, 12.51it/s][A
    Running loglikelihood requests:  20%|███████▌                              | 1583/7997 [05:30<08:06, 13.17it/s][A
    Running loglikelihood requests:  20%|███████▌                              | 1591/7997 [05:31<07:43, 13.82it/s][A
    Running loglikelihood requests:  20%|███████▌                              | 1599/7997 [05:31<07:50, 13.58it/s][A
    Running loglikelihood requests:  20%|███████▋                              | 1607/7997 [05:32<07:36, 13.99it/s][A
    Running loglikelihood requests:  20%|███████▋                              | 1615/7997 [05:32<07:37, 13.94it/s][A
    Running loglikelihood requests:  20%|███████▋                              | 1623/7997 [05:36<19:27,  5.46it/s][A
    Running loglikelihood requests:  20%|███████▊                              | 1631/7997 [05:38<22:12,  4.78it/s][A
    Running loglikelihood requests:  20%|███████▊                              | 1639/7997 [05:38<16:54,  6.27it/s][A
    Running loglikelihood requests:  21%|███████▊                              | 1647/7997 [05:39<13:46,  7.68it/s][A
    Running loglikelihood requests:  21%|███████▊                              | 1655/7997 [05:39<11:43,  9.02it/s][A
    Running loglikelihood requests:  21%|███████▉                              | 1665/7997 [05:40<09:13, 11.45it/s][A
    Running loglikelihood requests:  21%|███████▉                              | 1673/7997 [05:40<08:10, 12.88it/s][A
    Running loglikelihood requests:  21%|███████▉                              | 1681/7997 [05:41<07:29, 14.06it/s][A
    Running loglikelihood requests:  21%|████████                              | 1689/7997 [05:41<07:04, 14.85it/s][A
    Running loglikelihood requests:  21%|████████                              | 1697/7997 [05:42<06:41, 15.68it/s][A
    Running loglikelihood requests:  21%|████████                              | 1705/7997 [05:42<06:27, 16.23it/s][A
    Running loglikelihood requests:  21%|████████▏                             | 1713/7997 [05:43<06:34, 15.91it/s][A
    Running loglikelihood requests:  22%|████████▏                             | 1721/7997 [05:43<06:31, 16.01it/s][A
    Running loglikelihood requests:  22%|████████▏                             | 1729/7997 [05:44<06:27, 16.19it/s][A
    Running loglikelihood requests:  22%|████████▎                             | 1737/7997 [05:44<06:26, 16.20it/s][A
    Running loglikelihood requests:  22%|████████▎                             | 1745/7997 [05:45<06:11, 16.83it/s][A
    Running loglikelihood requests:  22%|████████▎                             | 1753/7997 [05:45<06:08, 16.93it/s][A
    Running loglikelihood requests:  22%|████████▎                             | 1761/7997 [05:46<06:23, 16.26it/s][A
    Running loglikelihood requests:  22%|████████▍                             | 1769/7997 [05:46<06:15, 16.59it/s][A
    Running loglikelihood requests:  22%|████████▍                             | 1777/7997 [05:46<06:02, 17.18it/s][A
    Running loglikelihood requests:  22%|████████▍                             | 1785/7997 [05:47<05:59, 17.28it/s][A
    Running loglikelihood requests:  22%|████████▌                             | 1793/7997 [05:47<05:54, 17.49it/s][A
    Running loglikelihood requests:  23%|████████▌                             | 1801/7997 [05:48<05:57, 17.32it/s][A
    Running loglikelihood requests:  23%|████████▌                             | 1809/7997 [05:48<05:59, 17.23it/s][A
    Running loglikelihood requests:  23%|████████▋                             | 1817/7997 [05:49<06:15, 16.46it/s][A
    Running loglikelihood requests:  23%|████████▋                             | 1825/7997 [05:49<06:20, 16.21it/s][A
    Running loglikelihood requests:  23%|████████▋                             | 1833/7997 [05:50<06:11, 16.58it/s][A
    Running loglikelihood requests:  23%|████████▋                             | 1841/7997 [05:50<06:05, 16.83it/s][A
    Running loglikelihood requests:  23%|████████▊                             | 1849/7997 [05:51<05:59, 17.09it/s][A
    Running loglikelihood requests:  23%|████████▊                             | 1857/7997 [05:51<05:54, 17.33it/s][A
    Running loglikelihood requests:  23%|████████▊                             | 1865/7997 [05:53<12:33,  8.14it/s][A
    Running loglikelihood requests:  23%|████████▉                             | 1873/7997 [05:55<13:45,  7.42it/s][A
    Running loglikelihood requests:  24%|████████▉                             | 1881/7997 [05:55<11:21,  8.97it/s][A
    Running loglikelihood requests:  24%|████████▉                             | 1889/7997 [05:56<09:43, 10.48it/s][A
    Running loglikelihood requests:  24%|█████████                             | 1897/7997 [05:56<08:37, 11.80it/s][A
    Running loglikelihood requests:  24%|█████████                             | 1905/7997 [05:57<08:12, 12.37it/s][A
    Running loglikelihood requests:  24%|█████████                             | 1913/7997 [05:57<07:51, 12.91it/s][A
    Running loglikelihood requests:  24%|█████████▏                            | 1921/7997 [05:58<07:08, 14.19it/s][A
    Running loglikelihood requests:  24%|█████████▏                            | 1929/7997 [05:58<07:28, 13.54it/s][A
    Running loglikelihood requests:  24%|█████████▏                            | 1937/7997 [05:59<07:29, 13.48it/s][A
    Running loglikelihood requests:  24%|█████████▏                            | 1945/7997 [05:59<07:37, 13.22it/s][A
    Running loglikelihood requests:  24%|█████████▎                            | 1953/7997 [06:00<07:00, 14.38it/s][A
    Running loglikelihood requests:  25%|█████████▎                            | 1961/7997 [06:00<06:33, 15.34it/s][A
    Running loglikelihood requests:  25%|█████████▎                            | 1969/7997 [06:01<06:37, 15.18it/s][A
    Running loglikelihood requests:  25%|█████████▍                            | 1977/7997 [06:01<06:18, 15.92it/s][A
    Running loglikelihood requests:  25%|█████████▍                            | 1985/7997 [06:02<06:18, 15.89it/s][A
    Running loglikelihood requests:  25%|█████████▍                            | 1993/7997 [06:02<06:19, 15.83it/s][A
    Running loglikelihood requests:  25%|█████████▌                            | 2001/7997 [06:07<20:37,  4.84it/s][A
    Running loglikelihood requests:  25%|█████████▌                            | 2009/7997 [06:09<24:56,  4.00it/s][A
    Running loglikelihood requests:  25%|█████████▌                            | 2017/7997 [06:10<19:50,  5.02it/s][A
    Running loglikelihood requests:  25%|█████████▌                            | 2025/7997 [06:11<15:31,  6.41it/s][A
    Running loglikelihood requests:  25%|█████████▋                            | 2033/7997 [06:11<12:27,  7.98it/s][A
    Running loglikelihood requests:  26%|█████████▋                            | 2041/7997 [06:11<10:18,  9.63it/s][A
    Running loglikelihood requests:  26%|█████████▋                            | 2049/7997 [06:12<09:25, 10.51it/s][A
    Running loglikelihood requests:  26%|█████████▊                            | 2057/7997 [06:13<08:24, 11.78it/s][A
    Running loglikelihood requests:  26%|█████████▊                            | 2065/7997 [06:13<07:47, 12.68it/s][A
    Running loglikelihood requests:  26%|█████████▊                            | 2073/7997 [06:13<07:04, 13.97it/s][A
    Running loglikelihood requests:  26%|█████████▉                            | 2081/7997 [06:14<06:30, 15.16it/s][A
    Running loglikelihood requests:  26%|█████████▉                            | 2089/7997 [06:14<06:19, 15.55it/s][A
    Running loglikelihood requests:  26%|█████████▉                            | 2097/7997 [06:15<06:03, 16.25it/s][A
    Running loglikelihood requests:  26%|██████████                            | 2105/7997 [06:15<05:47, 16.95it/s][A
    Running loglikelihood requests:  26%|██████████                            | 2113/7997 [06:16<05:57, 16.45it/s][A
    Running loglikelihood requests:  27%|██████████                            | 2121/7997 [06:16<05:42, 17.14it/s][A
    Running loglikelihood requests:  27%|██████████                            | 2129/7997 [06:17<06:12, 15.76it/s][A
    Running loglikelihood requests:  27%|██████████▏                           | 2137/7997 [06:17<05:59, 16.29it/s][A
    Running loglikelihood requests:  27%|██████████▏                           | 2145/7997 [06:18<06:23, 15.25it/s][A
    Running loglikelihood requests:  27%|██████████▏                           | 2153/7997 [06:18<06:23, 15.24it/s][A
    Running loglikelihood requests:  27%|██████████▎                           | 2161/7997 [06:19<07:15, 13.41it/s][A
    Running loglikelihood requests:  27%|██████████▎                           | 2169/7997 [06:20<06:51, 14.15it/s][A
    Running loglikelihood requests:  27%|██████████▎                           | 2177/7997 [06:20<06:28, 14.99it/s][A
    Running loglikelihood requests:  27%|██████████▍                           | 2185/7997 [06:21<06:55, 14.00it/s][A
    Running loglikelihood requests:  27%|██████████▍                           | 2193/7997 [06:21<06:49, 14.18it/s][A
    Running loglikelihood requests:  28%|██████████▍                           | 2201/7997 [06:22<06:27, 14.97it/s][A
    Running loglikelihood requests:  28%|██████████▍                           | 2209/7997 [06:22<06:16, 15.36it/s][A
    Running loglikelihood requests:  28%|██████████▌                           | 2218/7997 [06:23<06:18, 15.26it/s][A
    Running loglikelihood requests:  28%|██████████▌                           | 2226/7997 [06:23<05:55, 16.23it/s][A
    Running loglikelihood requests:  28%|██████████▌                           | 2234/7997 [06:24<05:38, 17.02it/s][A
    Running loglikelihood requests:  28%|██████████▋                           | 2242/7997 [06:24<05:46, 16.59it/s][A
    Running loglikelihood requests:  28%|██████████▋                           | 2250/7997 [06:25<05:54, 16.22it/s][A
    Running loglikelihood requests:  28%|██████████▋                           | 2258/7997 [06:25<05:43, 16.73it/s][A
    Running loglikelihood requests:  28%|██████████▊                           | 2266/7997 [06:26<06:08, 15.53it/s][A
    Running loglikelihood requests:  28%|██████████▊                           | 2274/7997 [06:26<06:11, 15.42it/s][A
    Running loglikelihood requests:  29%|██████████▊                           | 2282/7997 [06:27<06:24, 14.88it/s][A
    Running loglikelihood requests:  29%|██████████▉                           | 2290/7997 [06:27<06:31, 14.58it/s][A
    Running loglikelihood requests:  29%|██████████▉                           | 2298/7997 [06:28<06:15, 15.16it/s][A
    Running loglikelihood requests:  29%|██████████▉                           | 2306/7997 [06:28<06:15, 15.15it/s][A
    Running loglikelihood requests:  29%|██████████▉                           | 2314/7997 [06:29<05:56, 15.95it/s][A
    Running loglikelihood requests:  29%|███████████                           | 2322/7997 [06:29<05:44, 16.49it/s][A
    Running loglikelihood requests:  29%|███████████                           | 2330/7997 [06:30<06:00, 15.71it/s][A
    Running loglikelihood requests:  29%|███████████                           | 2338/7997 [06:31<06:20, 14.89it/s][A
    Running loglikelihood requests:  29%|███████████▏                          | 2346/7997 [06:31<06:07, 15.39it/s][A
    Running loglikelihood requests:  29%|███████████▏                          | 2357/7997 [06:32<07:01, 13.40it/s][A
    Running loglikelihood requests:  30%|███████████▏                          | 2365/7997 [06:33<06:58, 13.46it/s][A
    Running loglikelihood requests:  30%|███████████▎                          | 2373/7997 [06:33<07:14, 12.95it/s][A
    Running loglikelihood requests:  30%|███████████▎                          | 2382/7997 [06:34<07:08, 13.09it/s][A
    Running loglikelihood requests:  30%|███████████▎                          | 2393/7997 [06:35<06:28, 14.42it/s][A
    Running loglikelihood requests:  30%|███████████▍                          | 2401/7997 [06:35<06:22, 14.64it/s][A
    Running loglikelihood requests:  30%|███████████▍                          | 2409/7997 [06:36<06:48, 13.69it/s][A
    Running loglikelihood requests:  30%|███████████▍                          | 2417/7997 [06:36<06:49, 13.64it/s][A
    Running loglikelihood requests:  30%|███████████▌                          | 2425/7997 [06:37<07:06, 13.07it/s][A
    Running loglikelihood requests:  30%|███████████▌                          | 2433/7997 [06:38<07:31, 12.31it/s][A
    Running loglikelihood requests:  31%|███████████▌                          | 2441/7997 [06:38<06:48, 13.61it/s][A
    Running loglikelihood requests:  31%|███████████▋                          | 2449/7997 [06:39<06:20, 14.59it/s][A
    Running loglikelihood requests:  31%|███████████▋                          | 2457/7997 [06:39<06:17, 14.69it/s][A
    Running loglikelihood requests:  31%|███████████▋                          | 2465/7997 [06:40<05:57, 15.47it/s][A
    Running loglikelihood requests:  31%|███████████▊                          | 2473/7997 [06:40<06:13, 14.78it/s][A
    Running loglikelihood requests:  31%|███████████▊                          | 2481/7997 [06:41<05:49, 15.79it/s][A
    Running loglikelihood requests:  31%|███████████▊                          | 2489/7997 [06:41<05:47, 15.84it/s][A
    Running loglikelihood requests:  31%|███████████▊                          | 2497/7997 [06:42<05:33, 16.49it/s][A
    Running loglikelihood requests:  31%|███████████▉                          | 2505/7997 [06:42<05:33, 16.49it/s][A
    Running loglikelihood requests:  31%|███████████▉                          | 2513/7997 [06:42<05:16, 17.34it/s][A
    Running loglikelihood requests:  32%|███████████▉                          | 2521/7997 [06:43<05:06, 17.87it/s][A
    Running loglikelihood requests:  32%|████████████                          | 2529/7997 [06:43<05:01, 18.14it/s][A
    Running loglikelihood requests:  32%|████████████                          | 2541/7997 [06:44<04:15, 21.33it/s][A
    Running loglikelihood requests:  32%|████████████                          | 2550/7997 [06:44<04:25, 20.54it/s][A
    Running loglikelihood requests:  32%|████████████▏                         | 2558/7997 [06:45<04:27, 20.36it/s][A
    Running loglikelihood requests:  32%|████████████▏                         | 2572/7997 [06:45<03:45, 24.00it/s][A
    Running loglikelihood requests:  32%|████████████▎                         | 2580/7997 [06:45<03:56, 22.92it/s][A
    Running loglikelihood requests:  32%|████████████▎                         | 2588/7997 [06:46<04:21, 20.70it/s][A
    Running loglikelihood requests:  32%|████████████▎                         | 2596/7997 [06:46<04:22, 20.54it/s][A
    Running loglikelihood requests:  33%|████████████▎                         | 2604/7997 [06:47<04:30, 19.95it/s][A
    Running loglikelihood requests:  33%|████████████▍                         | 2612/7997 [06:47<04:42, 19.06it/s][A
    Running loglikelihood requests:  33%|████████████▍                         | 2620/7997 [06:48<04:40, 19.15it/s][A
    Running loglikelihood requests:  33%|████████████▍                         | 2628/7997 [06:48<04:53, 18.27it/s][A
    Running loglikelihood requests:  33%|████████████▌                         | 2636/7997 [06:49<04:44, 18.84it/s][A
    Running loglikelihood requests:  33%|████████████▌                         | 2644/7997 [06:49<04:42, 18.98it/s][A
    Running loglikelihood requests:  33%|████████████▌                         | 2652/7997 [06:49<04:42, 18.89it/s][A
    Running loglikelihood requests:  33%|████████████▋                         | 2660/7997 [06:50<04:39, 19.11it/s][A
    Running loglikelihood requests:  33%|████████████▋                         | 2668/7997 [06:50<04:49, 18.41it/s][A
    Running loglikelihood requests:  33%|████████████▋                         | 2676/7997 [06:51<04:46, 18.60it/s][A
    Running loglikelihood requests:  34%|████████████▊                         | 2684/7997 [06:51<04:41, 18.86it/s][A
    Running loglikelihood requests:  34%|████████████▊                         | 2692/7997 [06:52<04:44, 18.67it/s][A
    Running loglikelihood requests:  34%|████████████▊                         | 2700/7997 [06:52<04:43, 18.65it/s][A
    Running loglikelihood requests:  34%|████████████▊                         | 2708/7997 [06:52<04:55, 17.90it/s][A
    Running loglikelihood requests:  34%|████████████▉                         | 2716/7997 [06:53<05:00, 17.59it/s][A
    Running loglikelihood requests:  34%|████████████▉                         | 2724/7997 [06:53<04:53, 17.97it/s][A
    Running loglikelihood requests:  34%|████████████▉                         | 2732/7997 [06:54<04:49, 18.20it/s][A
    Running loglikelihood requests:  34%|█████████████                         | 2740/7997 [06:54<05:00, 17.51it/s][A
    Running loglikelihood requests:  34%|█████████████                         | 2748/7997 [06:55<04:54, 17.83it/s][A
    Running loglikelihood requests:  34%|█████████████                         | 2756/7997 [06:55<04:47, 18.20it/s][A
    Running loglikelihood requests:  35%|█████████████▏                        | 2764/7997 [06:56<04:43, 18.45it/s][A
    Running loglikelihood requests:  35%|█████████████▏                        | 2772/7997 [06:56<04:40, 18.64it/s][A
    Running loglikelihood requests:  35%|█████████████▏                        | 2780/7997 [06:57<05:40, 15.32it/s][A
    Running loglikelihood requests:  35%|█████████████▏                        | 2788/7997 [06:57<05:29, 15.79it/s][A
    Running loglikelihood requests:  35%|█████████████▎                        | 2796/7997 [06:58<05:10, 16.75it/s][A
    Running loglikelihood requests:  35%|█████████████▎                        | 2804/7997 [06:58<05:50, 14.83it/s][A
    Running loglikelihood requests:  35%|█████████████▎                        | 2812/7997 [06:59<06:48, 12.70it/s][A
    Running loglikelihood requests:  35%|█████████████▍                        | 2820/7997 [07:00<06:58, 12.37it/s][A
    Running loglikelihood requests:  35%|█████████████▍                        | 2828/7997 [07:00<06:52, 12.54it/s][A
    Running loglikelihood requests:  35%|█████████████▍                        | 2836/7997 [07:01<07:51, 10.94it/s][A
    Running loglikelihood requests:  36%|█████████████▌                        | 2844/7997 [07:02<08:19, 10.32it/s][A
    Running loglikelihood requests:  36%|█████████████▌                        | 2852/7997 [07:03<08:15, 10.37it/s][A
    Running loglikelihood requests:  36%|█████████████▌                        | 2860/7997 [07:04<07:31, 11.38it/s][A
    Running loglikelihood requests:  36%|█████████████▋                        | 2868/7997 [07:04<06:35, 12.97it/s][A
    Running loglikelihood requests:  36%|█████████████▋                        | 2876/7997 [07:04<05:51, 14.55it/s][A
    Running loglikelihood requests:  36%|█████████████▋                        | 2884/7997 [07:05<06:21, 13.39it/s][A
    Running loglikelihood requests:  36%|█████████████▋                        | 2892/7997 [07:05<05:35, 15.20it/s][A
    Running loglikelihood requests:  36%|█████████████▊                        | 2901/7997 [07:06<04:59, 17.03it/s][A
    Running loglikelihood requests:  36%|█████████████▊                        | 2911/7997 [07:06<04:25, 19.18it/s][A
    Running loglikelihood requests:  37%|█████████████▊                        | 2919/7997 [07:07<04:37, 18.31it/s][A
    Running loglikelihood requests:  37%|█████████████▉                        | 2927/7997 [07:07<04:42, 17.96it/s][A
    Running loglikelihood requests:  37%|█████████████▉                        | 2937/7997 [07:08<04:16, 19.75it/s][A
    Running loglikelihood requests:  37%|█████████████▉                        | 2945/7997 [07:08<04:23, 19.16it/s][A
    Running loglikelihood requests:  37%|██████████████                        | 2953/7997 [07:08<04:25, 19.02it/s][A
    Running loglikelihood requests:  37%|██████████████                        | 2961/7997 [07:09<04:55, 17.04it/s][A
    Running loglikelihood requests:  37%|██████████████                        | 2969/7997 [07:09<04:55, 17.03it/s][A
    Running loglikelihood requests:  37%|██████████████▏                       | 2977/7997 [07:10<04:59, 16.73it/s][A
    Running loglikelihood requests:  37%|██████████████▏                       | 2985/7997 [07:10<04:50, 17.28it/s][A
    Running loglikelihood requests:  37%|██████████████▏                       | 2993/7997 [07:11<05:07, 16.27it/s][A
    Running loglikelihood requests:  38%|██████████████▎                       | 3001/7997 [07:11<04:52, 17.06it/s][A
    Running loglikelihood requests:  38%|██████████████▎                       | 3009/7997 [07:12<04:38, 17.94it/s][A
    Running loglikelihood requests:  38%|██████████████▎                       | 3017/7997 [07:12<04:24, 18.82it/s][A
    Running loglikelihood requests:  38%|██████████████▎                       | 3025/7997 [07:13<04:13, 19.62it/s][A
    Running loglikelihood requests:  38%|██████████████▍                       | 3033/7997 [07:13<04:47, 17.29it/s][A
    Running loglikelihood requests:  38%|██████████████▍                       | 3041/7997 [07:14<04:45, 17.35it/s][A
    Running loglikelihood requests:  38%|██████████████▍                       | 3049/7997 [07:14<04:48, 17.17it/s][A
    Running loglikelihood requests:  38%|██████████████▌                       | 3057/7997 [07:19<17:13,  4.78it/s][A
    Running loglikelihood requests:  38%|██████████████▌                       | 3065/7997 [07:23<25:53,  3.17it/s][A
    Running loglikelihood requests:  38%|██████████████▌                       | 3073/7997 [07:25<24:04,  3.41it/s][A
    Running loglikelihood requests:  39%|██████████████▋                       | 3081/7997 [07:27<22:21,  3.67it/s][A
    Running loglikelihood requests:  39%|██████████████▋                       | 3089/7997 [07:28<18:01,  4.54it/s][A
    Running loglikelihood requests:  39%|██████████████▋                       | 3097/7997 [07:28<14:03,  5.81it/s][A
    Running loglikelihood requests:  39%|██████████████▊                       | 3105/7997 [07:29<11:14,  7.25it/s][A
    Running loglikelihood requests:  39%|██████████████▊                       | 3113/7997 [07:29<09:59,  8.15it/s][A
    Running loglikelihood requests:  39%|██████████████▊                       | 3116/7997 [07:29<08:59,  9.05it/s][A
    Running loglikelihood requests:  39%|██████████████▊                       | 3121/7997 [07:30<08:44,  9.30it/s][A
    Running loglikelihood requests:  39%|██████████████▊                       | 3130/7997 [07:30<06:47, 11.94it/s][A
    Running loglikelihood requests:  39%|██████████████▉                       | 3146/7997 [07:31<04:42, 17.16it/s][A
    Running loglikelihood requests:  39%|██████████████▉                       | 3154/7997 [07:31<04:28, 18.04it/s][A
    Running loglikelihood requests:  40%|███████████████                       | 3162/7997 [07:32<04:17, 18.75it/s][A
    Running loglikelihood requests:  40%|███████████████                       | 3170/7997 [07:32<04:12, 19.12it/s][A
    Running loglikelihood requests:  40%|███████████████                       | 3178/7997 [07:32<04:06, 19.52it/s][A
    Running loglikelihood requests:  40%|███████████████▏                      | 3186/7997 [07:33<04:13, 18.96it/s][A
    Running loglikelihood requests:  40%|███████████████▏                      | 3194/7997 [07:33<04:05, 19.54it/s][A
    Running loglikelihood requests:  40%|███████████████▏                      | 3202/7997 [07:34<04:05, 19.51it/s][A
    Running loglikelihood requests:  40%|███████████████▎                      | 3210/7997 [07:34<03:59, 19.99it/s][A
    Running loglikelihood requests:  40%|███████████████▎                      | 3218/7997 [07:34<04:16, 18.61it/s][A
    Running loglikelihood requests:  40%|███████████████▎                      | 3226/7997 [07:35<04:06, 19.38it/s][A
    Running loglikelihood requests:  40%|███████████████▎                      | 3234/7997 [07:35<04:13, 18.78it/s][A
    Running loglikelihood requests:  41%|███████████████▍                      | 3242/7997 [07:36<04:21, 18.18it/s][A
    Running loglikelihood requests:  41%|███████████████▍                      | 3250/7997 [07:36<04:24, 17.93it/s][A
    Running loglikelihood requests:  41%|███████████████▍                      | 3258/7997 [07:37<04:09, 19.00it/s][A
    Running loglikelihood requests:  41%|███████████████▌                      | 3266/7997 [07:37<04:11, 18.84it/s][A
    Running loglikelihood requests:  41%|███████████████▌                      | 3274/7997 [07:37<04:04, 19.34it/s][A
    Running loglikelihood requests:  41%|███████████████▌                      | 3282/7997 [07:38<04:14, 18.51it/s][A
    Running loglikelihood requests:  41%|███████████████▋                      | 3290/7997 [07:38<04:08, 18.98it/s][A
    Running loglikelihood requests:  41%|███████████████▋                      | 3298/7997 [07:39<04:01, 19.44it/s][A
    Running loglikelihood requests:  41%|███████████████▋                      | 3306/7997 [07:39<03:57, 19.72it/s][A
    Running loglikelihood requests:  41%|███████████████▋                      | 3314/7997 [07:39<04:09, 18.75it/s][A
    Running loglikelihood requests:  42%|███████████████▊                      | 3322/7997 [07:40<03:58, 19.64it/s][A
    Running loglikelihood requests:  42%|███████████████▊                      | 3330/7997 [07:40<03:52, 20.05it/s][A
    Running loglikelihood requests:  42%|███████████████▊                      | 3339/7997 [07:41<03:40, 21.15it/s][A
    Running loglikelihood requests:  42%|███████████████▉                      | 3347/7997 [07:41<03:42, 20.90it/s][A
    Running loglikelihood requests:  42%|███████████████▉                      | 3356/7997 [07:41<03:45, 20.56it/s][A
    Running loglikelihood requests:  42%|███████████████▉                      | 3367/7997 [07:42<03:22, 22.89it/s][A
    Running loglikelihood requests:  42%|████████████████                      | 3375/7997 [07:42<03:37, 21.28it/s][A
    Running loglikelihood requests:  42%|████████████████                      | 3383/7997 [07:43<03:36, 21.28it/s][A
    Running loglikelihood requests:  42%|████████████████                      | 3391/7997 [07:43<03:52, 19.84it/s][A
    Running loglikelihood requests:  43%|████████████████▏                     | 3399/7997 [07:44<03:51, 19.86it/s][A
    Running loglikelihood requests:  43%|████████████████▏                     | 3407/7997 [07:44<03:58, 19.25it/s][A
    Running loglikelihood requests:  43%|████████████████▏                     | 3415/7997 [07:44<03:58, 19.23it/s][A
    Running loglikelihood requests:  43%|████████████████▎                     | 3423/7997 [07:45<04:17, 17.78it/s][A
    Running loglikelihood requests:  43%|████████████████▎                     | 3431/7997 [07:45<04:04, 18.69it/s][A
    Running loglikelihood requests:  43%|████████████████▎                     | 3439/7997 [07:46<03:54, 19.44it/s][A
    Running loglikelihood requests:  43%|████████████████▍                     | 3447/7997 [07:46<03:49, 19.87it/s][A
    Running loglikelihood requests:  43%|████████████████▍                     | 3455/7997 [07:46<03:44, 20.27it/s][A
    Running loglikelihood requests:  43%|████████████████▍                     | 3463/7997 [07:47<03:58, 19.01it/s][A
    Running loglikelihood requests:  43%|████████████████▍                     | 3471/7997 [07:47<03:56, 19.12it/s][A
    Running loglikelihood requests:  44%|████████████████▌                     | 3479/7997 [07:48<03:50, 19.58it/s][A
    Running loglikelihood requests:  44%|████████████████▌                     | 3487/7997 [07:48<04:06, 18.29it/s][A
    Running loglikelihood requests:  44%|████████████████▌                     | 3495/7997 [07:49<03:55, 19.14it/s][A
    Running loglikelihood requests:  44%|████████████████▋                     | 3503/7997 [07:49<03:48, 19.70it/s][A
    Running loglikelihood requests:  44%|████████████████▋                     | 3511/7997 [07:49<03:45, 19.92it/s][A
    Running loglikelihood requests:  44%|████████████████▋                     | 3519/7997 [07:50<03:59, 18.68it/s][A
    Running loglikelihood requests:  44%|████████████████▊                     | 3527/7997 [07:50<03:45, 19.85it/s][A
    Running loglikelihood requests:  44%|████████████████▊                     | 3535/7997 [07:51<03:45, 19.83it/s][A
    Running loglikelihood requests:  44%|████████████████▊                     | 3543/7997 [07:51<03:39, 20.32it/s][A
    Running loglikelihood requests:  44%|████████████████▉                     | 3552/7997 [07:51<03:42, 20.01it/s][A
    Running loglikelihood requests:  45%|████████████████▉                     | 3562/7997 [07:52<03:24, 21.73it/s][A
    Running loglikelihood requests:  45%|████████████████▉                     | 3570/7997 [07:52<03:23, 21.75it/s][A
    Running loglikelihood requests:  45%|█████████████████                     | 3578/7997 [07:53<03:27, 21.31it/s][A
    Running loglikelihood requests:  45%|█████████████████                     | 3587/7997 [07:53<03:28, 21.20it/s][A
    Running loglikelihood requests:  45%|█████████████████                     | 3595/7997 [07:54<03:47, 19.36it/s][A
    Running loglikelihood requests:  45%|█████████████████▏                    | 3607/7997 [07:54<03:34, 20.49it/s][A
    Running loglikelihood requests:  45%|█████████████████▏                    | 3616/7997 [07:55<03:42, 19.70it/s][A
    Running loglikelihood requests:  45%|█████████████████▏                    | 3624/7997 [07:55<03:55, 18.56it/s][A
    Running loglikelihood requests:  45%|█████████████████▎                    | 3632/7997 [07:56<04:15, 17.08it/s][A
    Running loglikelihood requests:  46%|█████████████████▎                    | 3640/7997 [07:56<04:01, 18.01it/s][A
    Running loglikelihood requests:  46%|█████████████████▎                    | 3648/7997 [07:56<04:00, 18.06it/s][A
    Running loglikelihood requests:  46%|█████████████████▎                    | 3656/7997 [07:57<03:56, 18.32it/s][A
    Running loglikelihood requests:  46%|█████████████████▍                    | 3664/7997 [07:57<03:52, 18.67it/s][A
    Running loglikelihood requests:  46%|█████████████████▍                    | 3672/7997 [07:58<04:17, 16.80it/s][A
    Running loglikelihood requests:  46%|█████████████████▍                    | 3680/7997 [07:59<05:36, 12.83it/s][A
    Running loglikelihood requests:  46%|█████████████████▌                    | 3688/7997 [08:00<06:05, 11.78it/s][A
    Running loglikelihood requests:  46%|█████████████████▌                    | 3696/7997 [08:00<06:14, 11.48it/s][A
    Running loglikelihood requests:  46%|█████████████████▌                    | 3704/7997 [08:01<06:06, 11.72it/s][A
    Running loglikelihood requests:  46%|█████████████████▋                    | 3712/7997 [08:02<06:36, 10.80it/s][A
    Running loglikelihood requests:  46%|█████████████████▋                    | 3714/7997 [08:02<06:20, 11.26it/s][A
    Running loglikelihood requests:  47%|█████████████████▋                    | 3720/7997 [08:03<06:17, 11.34it/s][A
    Running loglikelihood requests:  47%|█████████████████▋                    | 3728/7997 [08:03<05:52, 12.09it/s][A
    Running loglikelihood requests:  47%|█████████████████▊                    | 3736/7997 [08:04<05:22, 13.22it/s][A
    Running loglikelihood requests:  47%|█████████████████▊                    | 3744/7997 [08:04<05:09, 13.76it/s][A
    Running loglikelihood requests:  47%|█████████████████▊                    | 3752/7997 [08:05<05:03, 13.98it/s][A
    Running loglikelihood requests:  47%|█████████████████▊                    | 3760/7997 [08:05<04:55, 14.33it/s][A
    Running loglikelihood requests:  47%|█████████████████▉                    | 3768/7997 [08:06<05:00, 14.08it/s][A
    Running loglikelihood requests:  47%|█████████████████▉                    | 3776/7997 [08:06<04:58, 14.12it/s][A
    Running loglikelihood requests:  47%|█████████████████▉                    | 3784/7997 [08:07<04:27, 15.78it/s][A
    Running loglikelihood requests:  47%|██████████████████                    | 3792/7997 [08:07<04:11, 16.70it/s][A
    Running loglikelihood requests:  48%|██████████████████                    | 3806/7997 [08:08<03:13, 21.67it/s][A
    Running loglikelihood requests:  48%|██████████████████▏                   | 3815/7997 [08:08<03:08, 22.21it/s][A
    Running loglikelihood requests:  48%|██████████████████▏                   | 3826/7997 [08:08<02:56, 23.64it/s][A
    Running loglikelihood requests:  48%|██████████████████▏                   | 3836/7997 [08:09<02:48, 24.67it/s][A
    Running loglikelihood requests:  48%|██████████████████▎                   | 3844/7997 [08:09<03:08, 22.00it/s][A
    Running loglikelihood requests:  48%|██████████████████▎                   | 3852/7997 [08:10<03:15, 21.15it/s][A
    Running loglikelihood requests:  48%|██████████████████▎                   | 3860/7997 [08:10<03:17, 20.99it/s][A
    Running loglikelihood requests:  48%|██████████████████▍                   | 3868/7997 [08:10<03:17, 20.88it/s][A
    Running loglikelihood requests:  48%|██████████████████▍                   | 3877/7997 [08:11<03:09, 21.72it/s][A
    Running loglikelihood requests:  49%|██████████████████▍                   | 3885/7997 [08:11<03:09, 21.75it/s][A
    Running loglikelihood requests:  49%|██████████████████▍                   | 3893/7997 [08:12<03:35, 19.06it/s][A
    Running loglikelihood requests:  49%|██████████████████▌                   | 3901/7997 [08:12<03:47, 18.00it/s][A
    Running loglikelihood requests:  49%|██████████████████▌                   | 3909/7997 [08:13<03:37, 18.78it/s][A
    Running loglikelihood requests:  49%|██████████████████▌                   | 3917/7997 [08:13<03:31, 19.33it/s][A
    Running loglikelihood requests:  49%|██████████████████▋                   | 3925/7997 [08:13<03:24, 19.88it/s][A
    Running loglikelihood requests:  49%|██████████████████▋                   | 3933/7997 [08:14<03:19, 20.34it/s][A
    Running loglikelihood requests:  49%|██████████████████▋                   | 3941/7997 [08:14<03:26, 19.67it/s][A
    Running loglikelihood requests:  49%|██████████████████▊                   | 3949/7997 [08:15<03:26, 19.64it/s][A
    Running loglikelihood requests:  49%|██████████████████▊                   | 3957/7997 [08:15<03:22, 19.98it/s][A
    Running loglikelihood requests:  50%|██████████████████▊                   | 3965/7997 [08:15<03:28, 19.30it/s][A
    Running loglikelihood requests:  50%|██████████████████▉                   | 3973/7997 [08:16<03:21, 19.97it/s][A
    Running loglikelihood requests:  50%|██████████████████▉                   | 3981/7997 [08:16<03:14, 20.61it/s][A
    Running loglikelihood requests:  50%|██████████████████▉                   | 3989/7997 [08:16<03:12, 20.80it/s][A
    Running loglikelihood requests:  50%|██████████████████▉                   | 3997/7997 [08:17<03:14, 20.60it/s][A
    Running loglikelihood requests:  50%|███████████████████                   | 4005/7997 [08:17<03:09, 21.07it/s][A
    Running loglikelihood requests:  50%|███████████████████                   | 4013/7997 [08:18<03:07, 21.25it/s][A
    Running loglikelihood requests:  50%|███████████████████                   | 4023/7997 [08:18<03:21, 19.72it/s][A
    Running loglikelihood requests:  50%|███████████████████▏                  | 4031/7997 [08:19<03:26, 19.17it/s][A
    Running loglikelihood requests:  51%|███████████████████▏                  | 4039/7997 [08:19<03:52, 17.04it/s][A
    Running loglikelihood requests:  51%|███████████████████▏                  | 4049/7997 [08:20<03:50, 17.11it/s][A
    Running loglikelihood requests:  51%|███████████████████▎                  | 4057/7997 [08:20<03:43, 17.62it/s][A
    Running loglikelihood requests:  51%|███████████████████▎                  | 4065/7997 [08:21<03:44, 17.48it/s][A
    Running loglikelihood requests:  51%|███████████████████▎                  | 4073/7997 [08:21<03:43, 17.56it/s][A
    Running loglikelihood requests:  51%|███████████████████▍                  | 4081/7997 [08:22<03:44, 17.43it/s][A
    Running loglikelihood requests:  51%|███████████████████▍                  | 4089/7997 [08:22<03:38, 17.88it/s][A
    Running loglikelihood requests:  51%|███████████████████▍                  | 4097/7997 [08:22<03:32, 18.37it/s][A
    Running loglikelihood requests:  51%|███████████████████▌                  | 4105/7997 [08:23<03:27, 18.78it/s][A
    Running loglikelihood requests:  51%|███████████████████▌                  | 4113/7997 [08:23<03:17, 19.66it/s][A
    Running loglikelihood requests:  52%|███████████████████▌                  | 4121/7997 [08:24<03:11, 20.22it/s][A
    Running loglikelihood requests:  52%|███████████████████▌                  | 4129/7997 [08:24<03:13, 20.04it/s][A
    Running loglikelihood requests:  52%|███████████████████▋                  | 4137/7997 [08:24<03:09, 20.32it/s][A
    Running loglikelihood requests:  52%|███████████████████▋                  | 4145/7997 [08:25<03:15, 19.73it/s][A
    Running loglikelihood requests:  52%|███████████████████▋                  | 4153/7997 [08:25<03:50, 16.66it/s][A
    Running loglikelihood requests:  52%|███████████████████▊                  | 4161/7997 [08:26<03:39, 17.49it/s][A
    Running loglikelihood requests:  52%|███████████████████▊                  | 4169/7997 [08:26<03:29, 18.26it/s][A
    Running loglikelihood requests:  52%|███████████████████▊                  | 4177/7997 [08:27<03:20, 19.03it/s][A
    Running loglikelihood requests:  52%|███████████████████▉                  | 4185/7997 [08:27<03:19, 19.10it/s][A
    Running loglikelihood requests:  52%|███████████████████▉                  | 4193/7997 [08:27<03:10, 19.93it/s][A
    Running loglikelihood requests:  53%|███████████████████▉                  | 4201/7997 [08:28<03:05, 20.42it/s][A
    Running loglikelihood requests:  53%|████████████████████                  | 4210/7997 [08:28<03:00, 20.93it/s][A
    Running loglikelihood requests:  53%|████████████████████                  | 4218/7997 [08:29<02:58, 21.20it/s][A
    Running loglikelihood requests:  53%|████████████████████                  | 4226/7997 [08:29<02:59, 21.06it/s][A
    Running loglikelihood requests:  53%|████████████████████▏                 | 4237/7997 [08:29<02:42, 23.17it/s][A
    Running loglikelihood requests:  53%|████████████████████▏                 | 4245/7997 [08:30<02:46, 22.52it/s][A
    Running loglikelihood requests:  53%|████████████████████▏                 | 4253/7997 [08:30<02:51, 21.80it/s][A
    Running loglikelihood requests:  53%|████████████████████▏                 | 4261/7997 [08:30<02:50, 21.88it/s][A
    Running loglikelihood requests:  53%|████████████████████▎                 | 4271/7997 [08:31<02:41, 23.12it/s][A
    Running loglikelihood requests:  54%|████████████████████▎                 | 4279/7997 [08:31<02:44, 22.62it/s][A
    Running loglikelihood requests:  54%|████████████████████▎                 | 4287/7997 [08:32<03:02, 20.31it/s][A
    Running loglikelihood requests:  54%|████████████████████▍                 | 4295/7997 [08:32<03:05, 19.95it/s][A
    Running loglikelihood requests:  54%|████████████████████▍                 | 4303/7997 [08:33<03:05, 19.94it/s][A
    Running loglikelihood requests:  54%|████████████████████▍                 | 4311/7997 [08:33<03:03, 20.10it/s][A
    Running loglikelihood requests:  54%|████████████████████▌                 | 4319/7997 [08:33<03:23, 18.10it/s][A
    Running loglikelihood requests:  54%|████████████████████▌                 | 4327/7997 [08:34<03:18, 18.52it/s][A
    Running loglikelihood requests:  54%|████████████████████▌                 | 4335/7997 [08:34<03:15, 18.76it/s][A
    Running loglikelihood requests:  54%|████████████████████▋                 | 4343/7997 [08:35<03:11, 19.10it/s][A
    Running loglikelihood requests:  54%|████████████████████▋                 | 4351/7997 [08:35<03:05, 19.60it/s][A
    Running loglikelihood requests:  55%|████████████████████▋                 | 4359/7997 [08:35<03:01, 20.00it/s][A
    Running loglikelihood requests:  55%|████████████████████▊                 | 4367/7997 [08:36<03:02, 19.92it/s][A
    Running loglikelihood requests:  55%|████████████████████▊                 | 4375/7997 [08:36<03:08, 19.20it/s][A
    Running loglikelihood requests:  55%|████████████████████▊                 | 4383/7997 [08:37<03:30, 17.14it/s][A
    Running loglikelihood requests:  55%|████████████████████▊                 | 4391/7997 [08:37<03:36, 16.69it/s][A
    Running loglikelihood requests:  55%|████████████████████▉                 | 4400/7997 [08:38<03:15, 18.41it/s][A
    Running loglikelihood requests:  55%|████████████████████▉                 | 4408/7997 [08:38<03:46, 15.84it/s][A
    Running loglikelihood requests:  55%|████████████████████▉                 | 4416/7997 [08:39<03:28, 17.19it/s][A
    Running loglikelihood requests:  55%|█████████████████████                 | 4424/7997 [08:39<03:18, 18.04it/s][A
    Running loglikelihood requests:  55%|█████████████████████                 | 4432/7997 [08:40<03:09, 18.83it/s][A
    Running loglikelihood requests:  56%|█████████████████████                 | 4440/7997 [08:40<03:04, 19.31it/s][A
    Running loglikelihood requests:  56%|█████████████████████▏                | 4450/7997 [08:40<02:53, 20.47it/s][A
    Running loglikelihood requests:  56%|█████████████████████▏                | 4462/7997 [08:41<02:26, 24.11it/s][A
    Running loglikelihood requests:  56%|█████████████████████▏                | 4470/7997 [08:41<02:34, 22.84it/s][A
    Running loglikelihood requests:  56%|█████████████████████▎                | 4478/7997 [08:42<02:38, 22.24it/s][A
    Running loglikelihood requests:  56%|█████████████████████▎                | 4486/7997 [08:42<02:40, 21.85it/s][A
    Running loglikelihood requests:  56%|█████████████████████▎                | 4494/7997 [08:42<02:40, 21.84it/s][A
    Running loglikelihood requests:  56%|█████████████████████▍                | 4502/7997 [08:43<02:42, 21.54it/s][A
    Running loglikelihood requests:  56%|█████████████████████▍                | 4510/7997 [08:43<02:47, 20.84it/s][A
    Running loglikelihood requests:  56%|█████████████████████▍                | 4518/7997 [08:43<02:44, 21.10it/s][A
    Running loglikelihood requests:  57%|█████████████████████▌                | 4526/7997 [08:44<02:46, 20.79it/s][A
    Running loglikelihood requests:  57%|█████████████████████▌                | 4534/7997 [08:44<02:50, 20.26it/s][A
    Running loglikelihood requests:  57%|█████████████████████▌                | 4542/7997 [08:45<02:49, 20.41it/s][A
    Running loglikelihood requests:  57%|█████████████████████▌                | 4550/7997 [08:45<02:44, 20.98it/s][A
    Running loglikelihood requests:  57%|█████████████████████▋                | 4558/7997 [08:45<02:48, 20.42it/s][A
    Running loglikelihood requests:  57%|█████████████████████▋                | 4566/7997 [08:46<02:50, 20.18it/s][A
    Running loglikelihood requests:  57%|█████████████████████▋                | 4574/7997 [08:46<02:39, 21.48it/s][A
    Running loglikelihood requests:  57%|█████████████████████▊                | 4582/7997 [08:47<02:35, 21.97it/s][A
    Running loglikelihood requests:  57%|█████████████████████▊                | 4590/7997 [08:47<02:32, 22.30it/s][A
    Running loglikelihood requests:  57%|█████████████████████▊                | 4598/7997 [08:47<02:31, 22.40it/s][A
    Running loglikelihood requests:  58%|█████████████████████▉                | 4609/7997 [08:48<02:13, 25.40it/s][A
    Running loglikelihood requests:  58%|█████████████████████▉                | 4619/7997 [08:48<02:06, 26.68it/s][A
    Running loglikelihood requests:  58%|█████████████████████▉                | 4627/7997 [08:48<02:17, 24.52it/s][A
    Running loglikelihood requests:  58%|██████████████████████                | 4636/7997 [08:49<02:11, 25.51it/s][A
    Running loglikelihood requests:  58%|██████████████████████                | 4647/7997 [08:49<02:00, 27.76it/s][A
    Running loglikelihood requests:  58%|██████████████████████▏               | 4658/7997 [08:49<01:54, 29.08it/s][A
    Running loglikelihood requests:  58%|██████████████████████▏               | 4666/7997 [08:50<02:04, 26.68it/s][A
    Running loglikelihood requests:  58%|██████████████████████▏               | 4678/7997 [08:50<01:53, 29.20it/s][A
    Running loglikelihood requests:  59%|██████████████████████▎               | 4686/7997 [08:50<01:58, 27.85it/s][A
    Running loglikelihood requests:  59%|██████████████████████▎               | 4697/7997 [08:51<01:54, 28.84it/s][A
    Running loglikelihood requests:  59%|██████████████████████▎               | 4705/7997 [08:51<01:59, 27.59it/s][A
    Running loglikelihood requests:  59%|██████████████████████▍               | 4713/7997 [08:51<02:02, 26.83it/s][A
    Running loglikelihood requests:  59%|██████████████████████▍               | 4721/7997 [08:52<02:10, 25.15it/s][A
    Running loglikelihood requests:  59%|██████████████████████▍               | 4729/7997 [08:52<02:13, 24.42it/s][A
    Running loglikelihood requests:  59%|██████████████████████▌               | 4737/7997 [08:52<02:13, 24.44it/s][A
    Running loglikelihood requests:  59%|██████████████████████▌               | 4745/7997 [08:53<02:13, 24.34it/s][A
    Running loglikelihood requests:  59%|██████████████████████▌               | 4753/7997 [08:53<02:15, 23.92it/s][A
    Running loglikelihood requests:  60%|██████████████████████▌               | 4761/7997 [08:53<02:30, 21.52it/s][A
    Running loglikelihood requests:  60%|██████████████████████▋               | 4769/7997 [08:54<02:25, 22.19it/s][A
    Running loglikelihood requests:  60%|██████████████████████▋               | 4777/7997 [08:54<02:30, 21.46it/s][A
    Running loglikelihood requests:  60%|██████████████████████▋               | 4785/7997 [08:55<02:59, 17.93it/s][A
    Running loglikelihood requests:  60%|██████████████████████▋               | 4787/7997 [08:55<03:05, 17.32it/s][A
    Running loglikelihood requests:  60%|██████████████████████▊               | 4794/7997 [08:56<03:20, 15.95it/s][A
    Running loglikelihood requests:  60%|██████████████████████▊               | 4806/7997 [08:56<02:43, 19.49it/s][A
    Running loglikelihood requests:  60%|██████████████████████▉               | 4814/7997 [08:56<02:46, 19.14it/s][A
    Running loglikelihood requests:  60%|██████████████████████▉               | 4824/7997 [08:57<02:35, 20.46it/s][A
    Running loglikelihood requests:  60%|██████████████████████▉               | 4832/7997 [08:57<02:35, 20.36it/s][A
    Running loglikelihood requests:  61%|██████████████████████▉               | 4840/7997 [08:58<02:28, 21.29it/s][A
    Running loglikelihood requests:  61%|███████████████████████               | 4848/7997 [08:58<03:08, 16.73it/s][A
    Running loglikelihood requests:  61%|███████████████████████               | 4856/7997 [08:59<03:13, 16.20it/s][A
    Running loglikelihood requests:  61%|███████████████████████▏              | 4868/7997 [08:59<02:41, 19.40it/s][A
    Running loglikelihood requests:  61%|███████████████████████▏              | 4880/7997 [09:00<02:29, 20.85it/s][A
    Running loglikelihood requests:  61%|███████████████████████▏              | 4888/7997 [09:00<02:31, 20.51it/s][A
    Running loglikelihood requests:  61%|███████████████████████▎              | 4896/7997 [09:01<02:35, 19.91it/s][A
    Running loglikelihood requests:  61%|███████████████████████▎              | 4904/7997 [09:01<02:44, 18.80it/s][A
    Running loglikelihood requests:  61%|███████████████████████▎              | 4912/7997 [09:01<02:41, 19.05it/s][A
    Running loglikelihood requests:  62%|███████████████████████▍              | 4920/7997 [09:02<02:35, 19.79it/s][A
    Running loglikelihood requests:  62%|███████████████████████▍              | 4928/7997 [09:02<02:28, 20.68it/s][A
    Running loglikelihood requests:  62%|███████████████████████▍              | 4936/7997 [09:03<02:23, 21.39it/s][A
    Running loglikelihood requests:  62%|███████████████████████▍              | 4944/7997 [09:10<16:25,  3.10it/s][A
    Running loglikelihood requests:  62%|███████████████████████▌              | 4952/7997 [09:12<15:22,  3.30it/s][A
    Running loglikelihood requests:  62%|███████████████████████▌              | 4960/7997 [09:13<11:11,  4.52it/s][A
    Running loglikelihood requests:  62%|███████████████████████▌              | 4971/7997 [09:13<07:23,  6.82it/s][A
    Running loglikelihood requests:  62%|███████████████████████▋              | 4979/7997 [09:13<05:59,  8.40it/s][A
    Running loglikelihood requests:  62%|███████████████████████▋              | 4989/7997 [09:14<04:38, 10.80it/s][A
    Running loglikelihood requests:  62%|███████████████████████▋              | 4997/7997 [09:14<03:56, 12.67it/s][A
    Running loglikelihood requests:  63%|███████████████████████▊              | 5009/7997 [09:14<02:58, 16.72it/s][A
    Running loglikelihood requests:  63%|███████████████████████▊              | 5019/7997 [09:15<02:36, 18.97it/s][A
    Running loglikelihood requests:  63%|███████████████████████▉              | 5027/7997 [09:15<02:27, 20.08it/s][A
    Running loglikelihood requests:  63%|███████████████████████▉              | 5035/7997 [09:16<02:52, 17.18it/s][A
    Running loglikelihood requests:  63%|███████████████████████▉              | 5043/7997 [09:16<02:39, 18.57it/s][A
    Running loglikelihood requests:  63%|████████████████████████              | 5051/7997 [09:16<02:36, 18.88it/s][A
    Running loglikelihood requests:  63%|████████████████████████              | 5059/7997 [09:17<02:33, 19.14it/s][A
    Running loglikelihood requests:  63%|████████████████████████              | 5067/7997 [09:17<02:25, 20.12it/s][A
    Running loglikelihood requests:  63%|████████████████████████▏             | 5078/7997 [09:17<02:02, 23.78it/s][A
    Running loglikelihood requests:  64%|████████████████████████▏             | 5086/7997 [09:18<02:23, 20.31it/s][A
    Running loglikelihood requests:  64%|████████████████████████▏             | 5094/7997 [09:19<02:35, 18.63it/s][A
    Running loglikelihood requests:  64%|████████████████████████▏             | 5102/7997 [09:19<02:35, 18.65it/s][A
    Running loglikelihood requests:  64%|████████████████████████▎             | 5110/7997 [09:19<02:30, 19.24it/s][A
    Running loglikelihood requests:  64%|████████████████████████▎             | 5118/7997 [09:20<02:31, 19.01it/s][A
    Running loglikelihood requests:  64%|████████████████████████▎             | 5126/7997 [09:20<02:25, 19.69it/s][A
    Running loglikelihood requests:  64%|████████████████████████▍             | 5134/7997 [09:21<02:22, 20.15it/s][A
    Running loglikelihood requests:  64%|████████████████████████▍             | 5142/7997 [09:21<02:23, 19.84it/s][A
    Running loglikelihood requests:  64%|████████████████████████▍             | 5150/7997 [09:21<02:28, 19.22it/s][A
    Running loglikelihood requests:  64%|████████████████████████▌             | 5158/7997 [09:22<02:19, 20.34it/s][A
    Running loglikelihood requests:  65%|████████████████████████▌             | 5168/7997 [09:22<02:06, 22.30it/s][A
    Running loglikelihood requests:  65%|████████████████████████▌             | 5181/7997 [09:22<01:43, 27.33it/s][A
    Running loglikelihood requests:  65%|████████████████████████▋             | 5190/7997 [09:23<01:42, 27.27it/s][A
    Running loglikelihood requests:  65%|████████████████████████▋             | 5200/7997 [09:23<01:39, 28.22it/s][A
    Running loglikelihood requests:  65%|████████████████████████▋             | 5208/7997 [09:23<01:41, 27.34it/s][A
    Running loglikelihood requests:  65%|████████████████████████▊             | 5218/7997 [09:24<01:38, 28.29it/s][A
    Running loglikelihood requests:  65%|████████████████████████▊             | 5227/7997 [09:24<01:39, 27.88it/s][A
    Running loglikelihood requests:  65%|████████████████████████▉             | 5235/7997 [09:24<01:43, 26.79it/s][A
    Running loglikelihood requests:  66%|████████████████████████▉             | 5243/7997 [09:25<01:45, 26.12it/s][A
    Running loglikelihood requests:  66%|████████████████████████▉             | 5252/7997 [09:25<01:56, 23.60it/s][A
    Running loglikelihood requests:  66%|█████████████████████████             | 5263/7997 [09:26<02:05, 21.70it/s][A
    Running loglikelihood requests:  66%|█████████████████████████             | 5271/7997 [09:26<02:18, 19.74it/s][A
    Running loglikelihood requests:  66%|█████████████████████████             | 5279/7997 [09:27<02:10, 20.83it/s][A
    Running loglikelihood requests:  66%|█████████████████████████             | 5287/7997 [09:27<02:07, 21.22it/s][A
    Running loglikelihood requests:  66%|█████████████████████████▏            | 5295/7997 [09:27<02:02, 22.12it/s][A
    Running loglikelihood requests:  66%|█████████████████████████▏            | 5303/7997 [09:31<07:36,  5.90it/s][A
    Running loglikelihood requests:  66%|█████████████████████████▏            | 5311/7997 [09:33<08:25,  5.31it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▎            | 5319/7997 [09:33<06:16,  7.10it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▎            | 5327/7997 [09:33<04:51,  9.16it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▎            | 5336/7997 [09:34<03:53, 11.42it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▍            | 5345/7997 [09:34<03:09, 14.01it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▍            | 5353/7997 [09:34<02:49, 15.62it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▍            | 5363/7997 [09:35<02:23, 18.36it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▌            | 5371/7997 [09:35<02:29, 17.62it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▌            | 5379/7997 [09:36<02:24, 18.08it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▌            | 5387/7997 [09:36<02:32, 17.09it/s][A
    Running loglikelihood requests:  67%|█████████████████████████▋            | 5395/7997 [09:37<02:18, 18.80it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▋            | 5405/7997 [09:37<02:07, 20.38it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▋            | 5417/7997 [09:37<01:47, 24.09it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▊            | 5425/7997 [09:38<01:45, 24.33it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▊            | 5433/7997 [09:38<01:46, 24.08it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▊            | 5444/7997 [09:38<01:38, 25.82it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▉            | 5455/7997 [09:39<01:33, 27.29it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▉            | 5463/7997 [09:39<01:38, 25.74it/s][A
    Running loglikelihood requests:  68%|█████████████████████████▉            | 5471/7997 [09:39<01:38, 25.55it/s][A
    Running loglikelihood requests:  69%|██████████████████████████            | 5479/7997 [09:40<01:40, 25.06it/s][A
    Running loglikelihood requests:  69%|██████████████████████████            | 5487/7997 [09:40<01:40, 25.00it/s][A
    Running loglikelihood requests:  69%|██████████████████████████            | 5495/7997 [09:40<01:43, 24.28it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▏           | 5503/7997 [09:41<01:44, 23.97it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▏           | 5511/7997 [09:41<01:47, 23.08it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▏           | 5519/7997 [09:41<01:45, 23.45it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▎           | 5529/7997 [09:42<01:38, 25.07it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▎           | 5537/7997 [09:42<01:40, 24.36it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▎           | 5545/7997 [09:43<01:42, 23.86it/s][A
    Running loglikelihood requests:  69%|██████████████████████████▍           | 5556/7997 [09:43<01:31, 26.59it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▍           | 5564/7997 [09:43<01:34, 25.65it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▍           | 5572/7997 [09:44<01:35, 25.30it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▌           | 5580/7997 [09:44<01:44, 23.05it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▌           | 5588/7997 [09:44<01:41, 23.72it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▌           | 5596/7997 [09:45<01:44, 23.03it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▋           | 5604/7997 [09:45<01:39, 24.12it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▋           | 5612/7997 [09:45<01:54, 20.81it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▋           | 5620/7997 [09:46<02:00, 19.73it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▋           | 5628/7997 [09:46<01:51, 21.21it/s][A
    Running loglikelihood requests:  70%|██████████████████████████▊           | 5636/7997 [09:47<01:55, 20.38it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▊           | 5644/7997 [09:47<01:53, 20.73it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▊           | 5652/7997 [09:47<01:49, 21.43it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▉           | 5660/7997 [09:48<01:54, 20.47it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▉           | 5668/7997 [09:48<01:49, 21.18it/s][A
    Running loglikelihood requests:  71%|██████████████████████████▉           | 5676/7997 [09:48<01:43, 22.49it/s][A
    Running loglikelihood requests:  71%|███████████████████████████           | 5684/7997 [09:49<01:52, 20.52it/s][A
    Running loglikelihood requests:  71%|███████████████████████████           | 5695/7997 [09:49<01:35, 24.08it/s][A
    Running loglikelihood requests:  71%|███████████████████████████           | 5705/7997 [09:50<01:29, 25.64it/s][A
    Running loglikelihood requests:  71%|███████████████████████████▏          | 5713/7997 [09:50<01:29, 25.60it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▏          | 5722/7997 [09:50<01:25, 26.70it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▏          | 5730/7997 [09:50<01:26, 26.33it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▎          | 5738/7997 [09:51<01:27, 25.95it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▎          | 5749/7997 [09:51<01:19, 28.23it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▎          | 5757/7997 [09:51<01:21, 27.52it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▍          | 5765/7997 [09:52<01:24, 26.30it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▍          | 5773/7997 [09:52<01:24, 26.40it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▍          | 5781/7997 [09:52<01:24, 26.18it/s][A
    Running loglikelihood requests:  72%|███████████████████████████▌          | 5789/7997 [09:53<01:26, 25.55it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▌          | 5799/7997 [09:53<01:19, 27.69it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▌          | 5807/7997 [09:53<01:23, 26.33it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▋          | 5818/7997 [09:54<01:26, 25.29it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▋          | 5826/7997 [09:54<01:32, 23.54it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▋          | 5834/7997 [09:55<01:29, 24.05it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▊          | 5842/7997 [09:55<01:33, 22.99it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▊          | 5850/7997 [09:55<01:31, 23.55it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▊          | 5858/7997 [09:56<01:29, 23.93it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▊          | 5866/7997 [10:04<11:56,  2.97it/s][A
    Running loglikelihood requests:  73%|███████████████████████████▉          | 5874/7997 [10:07<12:37,  2.80it/s][A
    Running loglikelihood requests:  74%|███████████████████████████▉          | 5882/7997 [10:08<09:49,  3.59it/s][A
    Running loglikelihood requests:  74%|███████████████████████████▉          | 5890/7997 [10:08<07:05,  4.96it/s][A
    Running loglikelihood requests:  74%|████████████████████████████          | 5900/7997 [10:08<04:49,  7.25it/s][A
    Running loglikelihood requests:  74%|████████████████████████████          | 5908/7997 [10:08<03:40,  9.47it/s][A
    Running loglikelihood requests:  74%|████████████████████████████          | 5916/7997 [10:09<03:02, 11.40it/s][A
    Running loglikelihood requests:  74%|████████████████████████████▏         | 5924/7997 [10:09<02:33, 13.53it/s][A
    Running loglikelihood requests:  74%|████████████████████████████▏         | 5935/7997 [10:09<01:59, 17.31it/s][A
    Running loglikelihood requests:  74%|████████████████████████████▏         | 5943/7997 [10:10<01:49, 18.72it/s][A
    Running loglikelihood requests:  74%|████████████████████████████▎         | 5951/7997 [10:10<01:40, 20.41it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▎         | 5961/7997 [10:10<01:27, 23.17it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▎         | 5969/7997 [10:11<01:26, 23.44it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▍         | 5977/7997 [10:11<01:24, 23.82it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▍         | 5985/7997 [10:11<01:22, 24.40it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▍         | 5993/7997 [10:12<01:21, 24.49it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▌         | 6002/7997 [10:12<01:21, 24.61it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▌         | 6010/7997 [10:12<01:20, 24.72it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▌         | 6018/7997 [10:13<01:24, 23.48it/s][A
    Running loglikelihood requests:  75%|████████████████████████████▋         | 6029/7997 [10:13<01:22, 23.96it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▋         | 6040/7997 [10:13<01:10, 27.79it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▋         | 6048/7997 [10:14<01:21, 24.02it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▊         | 6056/7997 [10:14<01:17, 24.92it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▊         | 6064/7997 [10:15<01:17, 25.01it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▊         | 6072/7997 [10:15<01:17, 24.96it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▉         | 6080/7997 [10:19<05:32,  5.77it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▉         | 6088/7997 [10:27<13:28,  2.36it/s][A
    Running loglikelihood requests:  76%|████████████████████████████▉         | 6096/7997 [10:33<16:55,  1.87it/s][A
    Running loglikelihood requests:  76%|█████████████████████████████         | 6104/7997 [10:37<16:00,  1.97it/s][A
    Running loglikelihood requests:  76%|█████████████████████████████         | 6112/7997 [10:40<14:22,  2.19it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████         | 6123/7997 [10:41<10:43,  2.91it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▏        | 6134/7997 [10:42<07:31,  4.12it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▏        | 6142/7997 [10:42<05:51,  5.27it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▏        | 6144/7997 [10:42<05:30,  5.61it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▏        | 6150/7997 [10:43<04:25,  6.94it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▎        | 6158/7997 [10:43<03:24,  8.98it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▎        | 6168/7997 [10:43<02:29, 12.26it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▎        | 6176/7997 [10:44<02:06, 14.37it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▍        | 6186/7997 [10:44<01:49, 16.60it/s][A
    Running loglikelihood requests:  77%|█████████████████████████████▍        | 6194/7997 [10:44<01:38, 18.37it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▍        | 6202/7997 [10:45<01:29, 19.98it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▌        | 6214/7997 [10:45<01:15, 23.57it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▌        | 6222/7997 [10:45<01:11, 24.96it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▌        | 6230/7997 [10:46<01:10, 25.14it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▋        | 6238/7997 [10:46<01:11, 24.71it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▋        | 6248/7997 [10:46<01:04, 27.11it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▋        | 6256/7997 [10:47<01:04, 26.95it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▊        | 6264/7997 [10:47<01:06, 25.91it/s][A
    Running loglikelihood requests:  78%|█████████████████████████████▊        | 6272/7997 [10:47<01:05, 26.28it/s][A
    Running loglikelihood requests:  79%|█████████████████████████████▊        | 6280/7997 [10:48<01:07, 25.44it/s][A
    Running loglikelihood requests:  79%|█████████████████████████████▉        | 6288/7997 [10:50<03:00,  9.48it/s][A
    Running loglikelihood requests:  79%|█████████████████████████████▉        | 6296/7997 [10:51<03:56,  7.20it/s][A
    Running loglikelihood requests:  79%|█████████████████████████████▉        | 6304/7997 [10:52<03:01,  9.30it/s][A
    Running loglikelihood requests:  79%|█████████████████████████████▉        | 6308/7997 [10:52<02:39, 10.56it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████        | 6315/7997 [10:52<02:11, 12.83it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████        | 6324/7997 [10:53<01:48, 15.38it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████        | 6333/7997 [10:53<01:28, 18.73it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████▏       | 6344/7997 [10:53<01:13, 22.41it/s][A
    Running loglikelihood requests:  79%|██████████████████████████████▏       | 6355/7997 [10:54<01:06, 24.83it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▎       | 6368/7997 [10:54<00:55, 29.52it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▎       | 6380/7997 [10:54<00:50, 32.20it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▎       | 6390/7997 [10:54<00:49, 32.23it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▍       | 6398/7997 [10:55<00:51, 30.88it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▍       | 6406/7997 [10:55<00:53, 29.67it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▍       | 6414/7997 [10:55<00:57, 27.43it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▌       | 6425/7997 [10:56<00:55, 28.47it/s][A
    Running loglikelihood requests:  80%|██████████████████████████████▌       | 6434/7997 [10:56<00:51, 30.13it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▌       | 6444/7997 [10:56<00:52, 29.85it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▋       | 6452/7997 [10:57<00:51, 29.93it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▋       | 6460/7997 [10:57<00:53, 28.85it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▋       | 6468/7997 [10:57<00:53, 28.53it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▊       | 6476/7997 [10:57<00:53, 28.38it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▊       | 6484/7997 [11:02<05:01,  5.01it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▊       | 6492/7997 [11:05<05:41,  4.41it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▉       | 6502/7997 [11:05<03:51,  6.46it/s][A
    Running loglikelihood requests:  81%|██████████████████████████████▉       | 6511/7997 [11:05<02:47,  8.88it/s][A
    Running loglikelihood requests:  82%|██████████████████████████████▉       | 6519/7997 [11:05<02:07, 11.57it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████       | 6527/7997 [11:05<01:41, 14.48it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████       | 6535/7997 [11:06<01:27, 16.79it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████       | 6543/7997 [11:06<01:17, 18.86it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▏      | 6551/7997 [11:06<01:09, 20.67it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▏      | 6562/7997 [11:06<00:58, 24.72it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▏      | 6570/7997 [11:07<00:56, 25.28it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▎      | 6581/7997 [11:07<00:49, 28.41it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▎      | 6589/7997 [11:07<00:49, 28.27it/s][A
    Running loglikelihood requests:  82%|███████████████████████████████▎      | 6597/7997 [11:08<00:50, 27.61it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▍      | 6605/7997 [11:08<00:51, 27.25it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▍      | 6613/7997 [11:08<00:57, 23.92it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▍      | 6625/7997 [11:09<00:46, 29.51it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▌      | 6633/7997 [11:09<00:48, 28.06it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▌      | 6641/7997 [11:09<00:51, 26.40it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▌      | 6649/7997 [11:10<00:52, 25.89it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▋      | 6658/7997 [11:10<00:51, 25.91it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▋      | 6666/7997 [11:10<00:57, 23.07it/s][A
    Running loglikelihood requests:  83%|███████████████████████████████▋      | 6674/7997 [11:11<00:57, 23.03it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▊      | 6682/7997 [11:11<00:53, 24.64it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▊      | 6690/7997 [11:11<00:51, 25.24it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▊      | 6698/7997 [11:15<03:19,  6.51it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▉      | 6713/7997 [11:16<02:45,  7.75it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▉      | 6724/7997 [11:16<01:58, 10.76it/s][A
    Running loglikelihood requests:  84%|███████████████████████████████▉      | 6732/7997 [11:17<01:36, 13.05it/s][A
    Running loglikelihood requests:  84%|████████████████████████████████      | 6743/7997 [11:17<01:18, 16.05it/s][A
    Running loglikelihood requests:  84%|████████████████████████████████      | 6753/7997 [11:17<01:03, 19.45it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▏     | 6762/7997 [11:18<00:57, 21.34it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▏     | 6772/7997 [11:18<00:55, 21.95it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▏     | 6782/7997 [11:18<00:50, 24.28it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▎     | 6790/7997 [11:19<00:51, 23.23it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▎     | 6798/7997 [11:19<00:50, 23.65it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▎     | 6801/7997 [11:19<00:49, 24.29it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▎     | 6808/7997 [11:19<00:51, 23.03it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▍     | 6816/7997 [11:20<00:51, 23.10it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▍     | 6825/7997 [11:20<00:47, 24.80it/s][A
    Running loglikelihood requests:  85%|████████████████████████████████▍     | 6834/7997 [11:20<00:43, 26.58it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▌     | 6845/7997 [11:21<00:39, 29.46it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▌     | 6856/7997 [11:21<00:37, 30.32it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▌     | 6864/7997 [11:21<00:40, 28.15it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▋     | 6872/7997 [11:22<00:41, 27.38it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▋     | 6880/7997 [11:31<06:49,  2.73it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▋     | 6890/7997 [11:34<05:48,  3.17it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▊     | 6898/7997 [11:34<04:40,  3.92it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▊     | 6906/7997 [11:34<03:25,  5.31it/s][A
    Running loglikelihood requests:  86%|████████████████████████████████▊     | 6914/7997 [11:35<02:31,  7.15it/s][A
    Running loglikelihood requests:  87%|████████████████████████████████▉     | 6925/7997 [11:35<01:44, 10.23it/s][A
    Running loglikelihood requests:  87%|████████████████████████████████▉     | 6937/7997 [11:35<01:15, 14.01it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████     | 6947/7997 [11:36<01:04, 16.36it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████     | 6958/7997 [11:36<00:51, 20.27it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████     | 6970/7997 [11:36<00:47, 21.55it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████▏    | 6978/7997 [11:37<00:45, 22.53it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████▏    | 6986/7997 [11:37<00:44, 22.48it/s][A
    Running loglikelihood requests:  87%|█████████████████████████████████▏    | 6995/7997 [11:37<00:41, 24.15it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▎    | 7008/7997 [11:38<00:36, 27.30it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▎    | 7016/7997 [11:38<00:36, 26.89it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▍    | 7024/7997 [11:38<00:37, 25.83it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▍    | 7032/7997 [11:39<00:38, 25.04it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▍    | 7040/7997 [11:39<00:38, 24.55it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▍    | 7048/7997 [11:42<02:11,  7.22it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▌    | 7056/7997 [11:46<04:01,  3.90it/s][A
    Running loglikelihood requests:  88%|█████████████████████████████████▌    | 7070/7997 [11:47<02:37,  5.88it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▋    | 7078/7997 [11:47<02:04,  7.38it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▋    | 7088/7997 [11:48<01:33,  9.72it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▋    | 7098/7997 [11:48<01:12, 12.48it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▊    | 7109/7997 [11:48<00:55, 15.87it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▊    | 7120/7997 [11:49<00:44, 19.73it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▉    | 7130/7997 [11:49<00:39, 21.92it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▉    | 7141/7997 [11:49<00:33, 25.40it/s][A
    Running loglikelihood requests:  89%|█████████████████████████████████▉    | 7150/7997 [11:49<00:31, 26.60it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████    | 7160/7997 [11:50<00:29, 28.43it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████    | 7173/7997 [11:50<00:24, 33.79it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████    | 7181/7997 [11:50<00:25, 32.35it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▏   | 7189/7997 [11:51<00:26, 30.53it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▏   | 7197/7997 [11:51<00:30, 26.52it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▏   | 7205/7997 [11:51<00:31, 24.96it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▎   | 7214/7997 [11:52<00:31, 25.18it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▎   | 7222/7997 [11:52<00:30, 25.78it/s][A
    Running loglikelihood requests:  90%|██████████████████████████████████▎   | 7231/7997 [11:52<00:28, 26.65it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▍   | 7239/7997 [11:53<00:27, 27.58it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▍   | 7249/7997 [11:53<00:26, 28.62it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▍   | 7259/7997 [11:53<00:23, 31.56it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▌   | 7270/7997 [11:54<00:23, 30.97it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▌   | 7281/7997 [11:54<00:20, 34.42it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▋   | 7289/7997 [11:54<00:22, 31.51it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▋   | 7302/7997 [11:54<00:18, 36.62it/s][A
    Running loglikelihood requests:  91%|██████████████████████████████████▋   | 7313/7997 [11:55<00:18, 37.09it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▊   | 7321/7997 [11:55<00:19, 34.28it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▊   | 7329/7997 [11:55<00:20, 32.80it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▊   | 7337/7997 [12:01<02:24,  4.58it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▉   | 7345/7997 [12:03<02:29,  4.35it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▉   | 7356/7997 [12:03<01:39,  6.47it/s][A
    Running loglikelihood requests:  92%|██████████████████████████████████▉   | 7364/7997 [12:03<01:14,  8.49it/s][A
    Running loglikelihood requests:  92%|███████████████████████████████████   | 7373/7997 [12:04<00:54, 11.43it/s][A
    Running loglikelihood requests:  92%|███████████████████████████████████   | 7385/7997 [12:04<00:38, 16.02it/s][A
    Running loglikelihood requests:  92%|███████████████████████████████████▏  | 7395/7997 [12:04<00:31, 19.15it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▏  | 7406/7997 [12:04<00:25, 23.14it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▏  | 7411/7997 [12:04<00:23, 25.36it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▏  | 7416/7997 [12:05<00:26, 21.86it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▎  | 7426/7997 [12:05<00:23, 24.24it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▎  | 7437/7997 [12:05<00:19, 28.17it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▍  | 7450/7997 [12:06<00:16, 33.89it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▍  | 7460/7997 [12:06<00:16, 32.87it/s][A
    Running loglikelihood requests:  93%|███████████████████████████████████▍  | 7469/7997 [12:06<00:16, 31.21it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▌  | 7478/7997 [12:07<00:16, 32.28it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▌  | 7486/7997 [12:07<00:17, 29.56it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▌  | 7497/7997 [12:07<00:15, 31.67it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▋  | 7505/7997 [12:08<00:15, 30.98it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▋  | 7519/7997 [12:08<00:13, 36.69it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▊  | 7532/7997 [12:08<00:11, 40.40it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▊  | 7540/7997 [12:08<00:11, 38.25it/s][A
    Running loglikelihood requests:  94%|███████████████████████████████████▉  | 7554/7997 [12:09<00:11, 39.33it/s][A
    Running loglikelihood requests:  95%|███████████████████████████████████▉  | 7566/7997 [12:09<00:10, 39.94it/s][A
    Running loglikelihood requests:  95%|███████████████████████████████████▉  | 7574/7997 [12:09<00:11, 35.40it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████  | 7584/7997 [12:09<00:11, 36.82it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████  | 7594/7997 [12:10<00:10, 38.47it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████▏ | 7608/7997 [12:10<00:09, 42.74it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████▏ | 7616/7997 [12:10<00:09, 38.54it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████▏ | 7624/7997 [12:11<00:10, 34.71it/s][A
    Running loglikelihood requests:  95%|████████████████████████████████████▎ | 7633/7997 [12:16<01:09,  5.21it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▎ | 7643/7997 [12:18<01:05,  5.37it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▎ | 7651/7997 [12:18<00:50,  6.88it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▍ | 7662/7997 [12:18<00:33,  9.99it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▍ | 7672/7997 [12:18<00:24, 13.21it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▍ | 7680/7997 [12:19<00:20, 15.24it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▌ | 7700/7997 [12:19<00:12, 24.49it/s][A
    Running loglikelihood requests:  96%|████████████████████████████████████▋ | 7713/7997 [12:19<00:09, 29.74it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▋ | 7722/7997 [12:19<00:08, 30.94it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▋ | 7733/7997 [12:20<00:07, 34.23it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▊ | 7744/7997 [12:20<00:06, 36.62it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▊ | 7755/7997 [12:20<00:06, 38.27it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▉ | 7770/7997 [12:20<00:05, 43.61it/s][A
    Running loglikelihood requests:  97%|████████████████████████████████████▉ | 7785/7997 [12:21<00:04, 48.06it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████ | 7802/7997 [12:21<00:04, 46.63it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████ | 7812/7997 [12:21<00:04, 43.34it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▏| 7820/7997 [12:22<00:04, 38.56it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▏| 7828/7997 [12:22<00:04, 38.08it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▏| 7837/7997 [12:22<00:04, 38.47it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▎| 7852/7997 [12:22<00:03, 45.27it/s][A
    Running loglikelihood requests:  98%|█████████████████████████████████████▍| 7870/7997 [12:23<00:02, 51.17it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▍| 7888/7997 [12:23<00:01, 57.23it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▌| 7905/7997 [12:23<00:01, 65.05it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▌| 7918/7997 [12:23<00:01, 61.15it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▋| 7926/7997 [12:23<00:01, 53.36it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▋| 7937/7997 [12:24<00:01, 49.50it/s][A
    Running loglikelihood requests:  99%|█████████████████████████████████████▊| 7949/7997 [12:24<00:01, 35.50it/s][A
    Running loglikelihood requests: 100%|█████████████████████████████████████▊| 7967/7997 [12:24<00:00, 47.68it/s][A
    Running loglikelihood requests: 100%|█████████████████████████████████████▉| 7976/7997 [12:28<00:02,  9.69it/s][A
    Running loglikelihood requests: 100%|██████████████████████████████████████| 7997/7997 [12:31<00:00, 10.64it/s][A
    Python(71492) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    fatal: not a git repository (or any of the parent directories): .git
    Python(71495) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(71496) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(71497) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(71498) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(71502) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(71503) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(71504) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(71505) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(71506) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    Python(71507) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.


    
    === w4a16 ===
    arc_easy {'acc,none': 0.551, 'acc_stderr,none': 0.016, 'acc_norm,none': 0.533, 'acc_norm_stderr,none': 0.016}
    hellaswag {'acc,none': 0.366, 'acc_stderr,none': 0.015, 'acc_norm,none': 0.458, 'acc_norm_stderr,none': 0.016}



```python
print(next(model.parameters()).device)   # cpu, mps:0 or cuda:0
#print(model.hf_device_map)
```

    mps:0



```python

```

# 使用 C# 和 ONNX 來玩 Phi-3 SLM

完整部落格文章連結：[https://blog.poychang.net/build-local-ai-chat-app-in-csharp-with-phi-3-mini-llm-and-onnx/](https://blog.poychang.net/build-local-ai-chat-app-in-csharp-with-phi-3-mini-llm-and-onnx/)

----------

LLM 洗捲世界對 AI 的認知之後，由於 LLM 需要的硬體要求實在太高，很難放到裝置上或落地運行，因此 SLM 逐漸受到重視，Phi-3 SLM 是由 Microsoft 所開發的模型，可以在你的電腦、手機等裝置來運行，這篇文章將帶你了解如何使用 C# 來操作 Phi-3 SLM 模型。

## 什麼是 Phi-3 SLM

Phi 系列的模型是由 Microsoft 所推出的 SLM (Small Language Model，小型語言模型)，而 Phi-3 SLM 則是目前最新的版本，強調在語言理解、推理、數學及寫程式等能力，且在執行效能與能力上做了一定程度的平衡，讓我們有機會能夠將語言模型放到使用者的裝置上運行。

Phi-3 也針對 ONNX Runtime 進行了優化，不僅支援在 Windows 執行，也能跨平台，甚至也針對 NVIDIA GPU 進行了優化，讓這個模型更加靈活且具可移植性。

目前 Phi-3 系列的模型有以下三種版本：

- Phi-3-mini (38 億參數)
- Phi3-small （70 億參數）
- Phi-3-medium （140 億參數）

關於他的效能比較，可以參考 Microsoft 所提供的下表或[此技術論文](https://arxiv.org/abs/2404.14219)：

![performance compare](https://i.imgur.com/5EbyJNg.png)

## 什麼是 ONNX Runtime

ONNX (Open Neural Network Exchange) 是一個開放的標準，用於操作機器學習模型，並在不同的框架間進行互操作。

ONNX Runtime 則是跨平台的機器學習模型加速器，具有彈性介面來整合硬體特定連結庫。ONNX Runtime 可以搭配來自 PyTorch、Tensorflow/Keras、TFLite、 scikit-learn 和其他架構的模型，更詳細資訊請參閱 [ONNX Runtime 文件](https://onnxruntime.ai/docs/)。

而關於 ONNX 的 .NET 函示庫，會有以下四個套件，分別的用途：

1. **Microsoft.ML.OnnxRuntimeGenAI**:
   - 這是 ONNX Runtime 的通用套件，包含執行 ONNX 模型所需的核心功能
   - 支援 CPU 執行，並且可以擴展支援其他硬體加速（例如 GPU）

2. **Microsoft.ML.OnnxRuntimeGenAI.Managed**:
   - 這是完全托管的版本，適用於純 .NET 環境
   - 不依賴原生程式庫，確保跨平台的一致性，適合在不需要特定硬體加速的情境下使用

3. **Microsoft.ML.OnnxRuntimeGenAI.Cuda**:
   - 這個版本專門針對使用 NVIDIA CUDA GPU 進行硬體加速
   - 適合需要高效能運算的深度學習模型，在 NVIDIA GPU 上可獲得顯著的性能提升

4. **Microsoft.ML.OnnxRuntimeGenAI.DirectML**:
   - 這個版本利用 Microsoft 的 DirectML API，專為 Windows 平台設計
   - 支援多種硬體加速裝置，包括 NVIDIA 和 AMD GPU，適用於 Windows 環境中的高效能運算需求

這些套件的主要差別在於它們針對不同的硬體加速需求和環境進行優化，選擇哪個套件取決於你的應用場景和硬體設置。一般來說，純 .NET 環境可使用 Managed 版本，如有 GPU 且需要用到 GPU 加速，則選擇 CUDA 或 DirectML 版本。

## 從 HuggingFace 下載 LLM 模型

目前 Phi-3 有以下幾種模型可以下載：

- Phi-3 Mini
  - Phi-3-mini-4k-instruct-onnx ([cpu, cuda, directml](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx))
  - Phi-3-mini-4k-instruct-onnx-web([web](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web))
  - Phi-3-mini-128k-instruct-onnx ([cpu, cuda, directml](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx))

- Phi-3 Small
  - Phi-3-small-8k-instruct-onnx ([cuda](https://huggingface.co/microsoft/Phi-3-small-8k-instruct-onnx-cuda))
  - Phi-3-small-128k-instruct-onnx ([cuda](https://huggingface.co/microsoft/Phi-3-small-128k-instruct-onnx-cuda))

- Phi-3 Medium
  - Phi-3-medium-4k-instruct-onnx ([cpu](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cpu), [cuda](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cuda), [directml](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-directml))
  - Phi-3-medium-128k-instruct-onnx ([cpu](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-cpu), [cuda](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-cuda), [directml](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-directml))

上面的模型名稱中，會標註 4k 和 128k，這是表示能組成上下文的 Token 長度，意思就是運行 4k 的模型所需要的資源較少，而 128k 則是能支援更大的上下文長度。

我們可以簡單把 HuggingFace 當成一個像是 GitHub 的地方，裡面存放著各種 Model 的資源，我們可以透過 git 的指令來下載 HuggingFace 上的模型。

下載之前，請先確認你的環境有安裝 [git-lfs](https://git-lfs.com)，你可以使用 `git lfs install` 指令進行安裝。

假設我們要下載的模型是 [microsoft/Phi-3-mini-4k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx)，下載指令就會試如下：

```bash
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
```

這篇文章的範例中，我們主要是會用到 `\Phi-3-mini-4k-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32` 資料夾底下的 `.onnx` 模型檔案，這個資料夾下的模型是支援僅用 CPU 運行的。

## 使用 C# 來建置小型聊天

建立完 .NET 的 Console App 專案後，我們需要先安裝 ONNX Runtime 的相關套件，這裡我們會使用 `Microsoft.ML.OnnxRuntime`、`Microsoft.ML.OnnxRuntimeGenAI`、`Microsoft.ML.OnnxRuntimeGenAI.Managed` 這三個套件。

```xml
<Project Sdk="Microsoft.NET.Sdk">

    <PropertyGroup>
        <OutputType>Exe</OutputType>
        <TargetFramework>net8.0</TargetFramework>
        <ImplicitUsings>enable</ImplicitUsings>
        <Nullable>enable</Nullable>
    </PropertyGroup>

    <ItemGroup>
        <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.18.0" />
        <PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI" Version="0.2.0" />
        <PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI.Managed" Version="0.2.0" />
    </ItemGroup>

</Project>
```

接著在 `Program.cs` 中加入以下程式碼，這會模擬使用者和 AI 一問一答：

```csharp
using Microsoft.ML.OnnxRuntimeGenAI;

// 提供存放 Phi-3 的 ONNX 模型資料夾位置，該資料夾內必須要有 .onnx 檔案
var modelPath = "C:\\Users\\poypo\\Code\\onnx\\Phi-3-mini-4k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32";
var model = new Model(modelPath);
var tokenizer = new Tokenizer(model);

// 設定 System Prompt 提示 AI 如何回答 User Prompt
var systemPrompt = "You are a knowledgeable and friendly assistant. Answer the following question as clearly and concisely as possible, providing any relevant information and examples.";

Console.WriteLine("Type Prompt then Press [Enter] or CTRL-C to Exit");
Console.WriteLine("");

// 模擬使用者和 AI　一問一答
while (true)
{
    // 取得 User Prompt
    Console.Write("User: ");
    var userPrompt = Console.ReadLine();

    // 組合 Prompt：將 System Prompt 和 User Prompt 組合在一起
    var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{userPrompt}<|end|><|assistant|>";

    // 將 Prompt 編碼成 Token
    var tokens = tokenizer.Encode(fullPrompt);

    // 設定生成器參數，完整參數列表請參考： https://onnxruntime.ai/docs/genai/reference/config.html
    var generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 2048);
    generatorParams.SetSearchOption("temperature", 0.3);
    generatorParams.SetInputSequences(tokens);

    // 產生回應
    Console.Write("Assistant: ");
    var generator = new Generator(model, generatorParams);
    // 將生成的每個 Token 逐一解碼成文字並輸出回應
    while (!generator.IsDone())
    {
        generator.ComputeLogits();
        generator.GenerateNextToken();
        var outputTokens = generator.GetSequence(0);
        var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
        var output = tokenizer.Decode(newToken);
        Console.Write(output);
    }
    Console.WriteLine();
}
```

程式碼中，主要分成三個部分：

1. 載入模型：透過 `Model` 類別來載入模型，並透過 `Tokenizer` 類別來將文字轉換成 Token
2. 設定 Prompt：設定 System Prompt 和 User Prompt，並將兩者組合成完整的 Prompt
3. 一問一答：透過 `Generator` 類別來生成回應，並將生成的 Token 解碼成文字輸出

在設定 `Generator` 的參數時，會透過 `GeneratorParams` 類別來做設定，這裡我們只設定了 `max_length` 和 `temperature` 兩個參數，`max_length` 是生成回應的最大長度，`temperature` 則是控制生成回應的多樣性。而 ONNX Runtime 所提供的設定參數還相當多，完整參數列表請參考 [官方文件](https://onnxruntime.ai/docs/genai/reference/config.html)。

這個小型聊天的運行效果如下：

![和 SLM 對話的畫面](https://i.imgur.com/vcJQQQD.png)

> 本篇完整範例程式碼請參考 [poychang/Phi3MiniConsoleApp](https://github.com/poychang/Phi3MiniConsoleApp)。

## 後記

測試起來，用英文對話可以很正確的回答問題，但改用中文的時候，會有許多奇妙的狀況發生。不過這也是可以預期的，畢竟這種 SLM 模型，基本上都是使用英文作為訓練材料，所以對於中文的處理能力就會比較弱。或許之後可以透過 Fine-tuning 的方式，來提升中文的處理能力，可以再研究看看。

---

參考資料：

* [Hugging Face - Downloading models](https://huggingface.co/docs/hub/models-downloading)
* [Build a Generative AI App in C# with Phi-3 SLM and ONNX](https://build5nines.com/build-a-generative-ai-app-in-c-with-phi-3-mini-llm-and-onnx/)
* [GitHub - microsoft/Phi-3CookBook](https://github.com/microsoft/Phi-3CookBook)
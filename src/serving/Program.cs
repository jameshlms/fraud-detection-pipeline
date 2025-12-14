using FraudDetectionService.models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SessionOptions = Microsoft.ML.OnnxRuntime.SessionOptions;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton(sp =>
{
    var env = sp.GetRequiredService<IHostEnvironment>();
    var modelPath = Path.Combine(env.ContentRootPath, "onnx/models/fraud_detection_xgbclassifier.onnx");

    var opts = new SessionOptions
    {
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
    };

    return new InferenceSession(modelPath, opts);
});

var app = builder.Build();

app.Map("/", () => "Fraud Detection ONNX Model Serving");
app.MapPost("/predict", (InferenceRequest request, InferenceSession session, ILogger<Program> logger) =>
{
    if (request.Inputs.Count == 0) return Results.BadRequest("No input data provided.");

    logger.LogInformation("Received inference request with {Count} inputs.", request.Inputs.Count);

    var inputCount = request.Inputs.Count;
    var featureCount = request.Inputs[0].Length;

    if (featureCount != 31)
    {
        logger.LogWarning("Invalid feature count: {FeatureCount}. Expected 31 features.", featureCount);
        return Results.BadRequest($"Invalid feature count: {featureCount}. Expected 31 features.");
    }

    var tensor = new DenseTensor<float>(new[] { inputCount, featureCount });

    logger.LogInformation("Preparing input tensor with shape [{InputCount}, {FeatureCount}].", inputCount,
        featureCount);

    for (var i = 0; i < inputCount; i++)
    {
        for (var j = 0; j < featureCount; j++)
        {
            tensor[i, j] = request.Inputs[i][j];
        }
    }

    var inputTensor = NamedOnnxValue.CreateFromTensor("input", tensor);
    logger.LogInformation("Created input tensor for inference.");

    var results = session.Run([inputTensor]);
    logger.LogInformation("Inference completed. Processing results.");

    var outputTensor = results[0].AsTensor<long>();
    var predictions = outputTensor.Select(v => (int)v).ToArray();
    logger.LogInformation("Processed {Count} predictions.", predictions.Length);

    return Results.Ok(predictions);
});

app.Run();

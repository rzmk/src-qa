use langchain_rust::{
    chain::{Chain, StuffDocument},
    document_loaders::{CsvLoader, Loader, SourceCodeLoader},
    llm::{OpenAI, OpenAIConfig},
};

use futures_util::StreamExt;

#[tokio::main]
async fn main() {
    let loader_with_dir = SourceCodeLoader::from_path("./src".to_string()).with_dir_loader_options(
        DirLoaderOptions {
            glob: None,
            suffixes: Some(vec!["rs".to_string()]),
            exclude: None,
        },
    );

    let stream = loader_with_dir.load().await.unwrap();
    let documents = stream.map(|x| x.unwrap()).collect::<Vec<_>>().await;

    // Since Ollama is OpenAI compatible
    // You can call Ollama this way:
    let llm = OpenAI::default()
        .with_config(
            OpenAIConfig::default()
                .with_api_base("http://localhost:11434/v1")
                .with_api_key("ollama"),
        )
        .with_model("llama3:8b-instruct-q4_K_M");

    let chain = StuffDocument::load_stuff_qa(llm);
    let input = chain
        .qa_prompt_builder()
        // You could also get the documents form a retriver
        .documents(&documents)
        .question("What's the code about?")
        .build();

    let output = chain.invoke(input).await.unwrap();

    println!("{}", output);
}

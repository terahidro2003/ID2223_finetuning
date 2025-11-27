import gradio as gr
from dotenv import load_dotenv
from typing import List, Dict

from src.llm import initialize_providers
from src.search.searcher import DuckDuckGoSearcher
from src.search.query_generator import QueryGenerator

# Load environment variables
load_dotenv()

# Global state
providers = initialize_providers()
if not providers:
    raise RuntimeError(
        "No LLM providers configured!\n"
        "Add API keys to .env:\n"
        "- <PROVIDER>_API_KEY\n"
        "- run llm locally & configure url"
    )
searcher = DuckDuckGoSearcher()


def format_search_results(results: Dict) -> str:
    """Format search results as markdown"""
    if not results or not results.get("results"):
        return ""
    
    md = f"### 🔍 Found {len(results['results'])} sources\n\n"
    
    if results.get("queries_used"):
        md += f"**Queries:** {', '.join(results['queries_used'])}\n\n---\n\n"
    
    for idx, result in enumerate(results["results"][:8], 1):
        md += f"**{idx}. [{result['title']}]({result['url']})**\n\n"
        if result.get("query_used"):
            md += f"*Via: {result['query_used']}*\n\n"
        md += f"{result['content'][:250]}...\n\n"
        if idx < len(results["results"]):
            md += "---\n\n"
    
    return md

def extract_text_from_message(message) -> str:
    if isinstance(message, str):
        return message
    elif isinstance(message, list):
        # Extract text from content blocks
        text_parts = []
        for block in message:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
        return " ".join(text_parts)
    else:
        return str(message)


def normalize_history(history: List[Dict]) -> List[Dict]:
    normalized = []
    for msg in history:
        normalized.append({
            "role": msg["role"],
            "content": extract_text_from_message(msg.get("content", ""))
        })
    return normalized


def respond(
    message,  # Can be string OR list of content blocks
    history: List[Dict],
    provider_name: str,
    model: str,
    temperature: float,
    enable_search: bool,
    auto_search: bool,
    num_queries: int
):
    # Extract text from message (handles both formats)
    user_text = extract_text_from_message(message)
    
    if not user_text.strip():
        yield history, ""
        return
    
    if provider_name not in providers:
        error_history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": f"❌ Provider '{provider_name}' not available"}
        ]
        yield error_history, ""
        return
    
    provider = providers[provider_name]
    search_results = None
    search_display = ""
    normalized_history = normalize_history(history)
    
    # Perform search if enabled
    if enable_search and user_text:
        try:
            messages = normalized_history + [{"role": "user", "content": user_text}]
            query_gen = QueryGenerator(provider, model)
            
            if auto_search and query_gen.should_search(user_text):
                queries = query_gen.generate_queries(user_text, num_queries)
                search_results = searcher.multi_query_search(queries, max_results_per_query=3)
                search_display = format_search_results(search_results)
        except Exception as e:
            print(f"Search error: {e}")
    
    # Build messages for LLM
    messages = normalized_history.copy()
    
    # Add search context
    if search_results and search_results.get("results"):
        context = "Web search results:\n\n"
        for idx, result in enumerate(search_results["results"][:5], 1):
            context += f"{idx}. {result['title']}\n"
            context += f"   {result['url']}\n"
            context += f"   {result['content'][:300]}...\n\n"
        context += "\nUse this information to answer. Cite sources when appropriate."
        
        messages.append({"role": "system", "content": context})
    
    messages.append({"role": "user", "content": user_text})
    
    # Get LLM response
    try:
        response_stream = provider.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=True
        )
        
        # Stream the response - yield full history each time
        full_response = ""
        for chunk in response_stream:
            full_response += chunk
            
            # Yield the complete updated history
            updated_history = history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": full_response}
            ]
            yield updated_history, search_display
            
    except Exception as e:
        error_history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": f"❌ Error: {str(e)}"}
        ]
        yield error_history, search_display


def update_models(provider_name: str):
    """Update model choices when provider changes"""
    if provider_name in providers:
        models = providers[provider_name].get_available_models()
        return gr.Dropdown(choices=models, value=models[0] if models else None)
    return gr.Dropdown(choices=[], value=None)


initial_provider = list(providers.keys())[0]
initial_models = providers[initial_provider].get_available_models()
# Build UI
with gr.Blocks(title="Multi-LLM Search") as demo:
    gr.Markdown("# 🤖 Multi-LLM Search Interface")
    gr.Markdown("Chat with multiple LLM providers + intelligent web search")
    
    with gr.Row():
        # Left: Chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=600,
                placeholder="<strong>Ask me anything!</strong><br>I can search the web for current information."
            )
            
            msg = gr.Textbox(
                label="Message",
                placeholder="Ask me anything...",
                lines=2
            )
            
            with gr.Row():
                clear = gr.Button("Clear")
                submit = gr.Button("Send", variant="primary")
            
            search_output = gr.Markdown(label="Search Results", visible=True)
        
        # Right: Settings
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            provider_dropdown = gr.Dropdown(
                choices=list(providers.keys()),
                value=initial_provider,
                label="Provider"
            )
            
            model_dropdown = gr.Dropdown(
                choices=initial_models,
                value=initial_models[0] if initial_models else None,
                label="Model"
            )
            
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            
            gr.Markdown("### 🔍 Search")
            
            enable_search = gr.Checkbox(value=True, label="Enable search")
            auto_search = gr.Checkbox(value=True, label="Auto-detect")
            num_queries = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Search queries"
            )
            
            gr.Markdown("### 📋 Examples")
            gr.Examples(
                examples=[
                    "What are the latest AI developments in 2025?",
                    "Explain quantum computing simply",
                    "What's the weather in Stockholm?"
                ],
                inputs=msg
            )
    
    # Event handlers
    submit_event = submit.click(
        fn=respond,
        inputs=[
            msg,
            chatbot,
            provider_dropdown,
            model_dropdown,
            temperature_slider,
            enable_search,
            auto_search,
            num_queries
        ],
        outputs=[chatbot, search_output]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    msg.submit(
        fn=respond,
        inputs=[
            msg,
            chatbot,
            provider_dropdown,
            model_dropdown,
            temperature_slider,
            enable_search,
            auto_search,
            num_queries
        ],
        outputs=[chatbot, search_output]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    clear.click(lambda: ([], ""), outputs=[chatbot, search_output])
    
    provider_dropdown.change(
        fn=update_models,
        inputs=[provider_dropdown],
        outputs=[model_dropdown]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)

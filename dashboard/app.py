import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import glob

# Configure the Streamlit page with a custom title and wide layout
st.set_page_config(page_title="BOLT⚡️ Leaderboard", layout="wide")

def load_results(results_file="results_final/results.json"):
    """Load evaluation results from the consolidated JSON file.
    
    Args:
        results_file (str): Path to the JSON file containing evaluation results
        
    Returns:
        dict: Loaded results containing model metrics and evaluations
    """
    with open(results_file, 'r') as f:
        return json.load(f)

def create_results_df(results):
    """Convert nested results dictionary to a flattened pandas DataFrame.
    
    Args:
        results (dict): Nested dictionary containing model evaluation results
        
    Returns:
        pd.DataFrame: Flattened DataFrame with columns for model name, variant,
                     and evaluation metrics (GPT score, syntax score, memory usage, etc.)
    """
    flat_data = []
    
    for model in results['models']:
        base_model_name = model['name']
        
        # Handle models without variants (like DeepSeek-Coder-7B)
        if 'variants' not in model:
            flat_result = {
                'model_name': base_model_name,
                'variant': 'base',
                'gpt_eval_score': model['metrics']['gpt_eval'],
                'syntax_eval_score': model['metrics']['syntax_eval'],
                'memory_mb': model['metrics']['memory'],
                'wall_time_seconds': model['metrics']['wall_time']
            }
            flat_data.append(flat_result)
            continue
            
        # Handle models with different fine-tuning variants
        for variant in model['variants']:
            flat_result = {
                'model_name': base_model_name,
                'variant': variant['type'],
                'gpt_eval_score': variant['metrics']['gpt_eval'],
                'syntax_eval_score': variant['metrics']['syntax_eval'],
                'memory_mb': variant['metrics']['memory'],
                'wall_time_seconds': variant['metrics']['wall_time']
            }
            flat_data.append(flat_result)
            
    return pd.DataFrame(flat_data)

def main():
    """Main function to render the BOLT leaderboard dashboard.
    
    Displays:
    - Interactive filters for models and variants
    - Resource usage analysis plots
    - Model performance comparisons
    - Leaderboard ranking
    - Performance vs resource trade-off analysis
    """
    st.title("BOLT⚡️ Leaderboard")
    
    # Load and validate results
    results = load_results()
    if not results:
        st.error("No results found in the results file!")
        return
    
    df = create_results_df(results)
    
    # Sidebar filters for interactive visualization
    st.sidebar.header("Filters")
    
    # Allow users to select specific models to compare
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=df['model_name'].unique(),
        default=df['model_name'].unique()
    )
    
    # Filter dataframe based on model selection
    filtered_df = df[df['model_name'].isin(selected_models)]
    
    # Allow users to select specific variants to compare
    selected_variants = st.sidebar.multiselect(
        "Select Variants",
        options=df['variant'].unique(),
        default=df['variant'].unique()
    )
    
    filtered_df = filtered_df[filtered_df['variant'].isin(selected_variants)]
    
    # Resource Usage Analysis
    st.subheader("Resource Usage Analysis")
    fig_memory_time = px.scatter(
        filtered_df,
        x='wall_time_seconds',
        y='memory_mb',
        color='model_name',
        symbol='variant',
        size='gpt_eval_score',
        hover_data=['syntax_eval_score'],
        title='Memory vs Wall Time Usage',
        labels={
            'wall_time_seconds': 'Wall Time (seconds)',
            'memory_mb': 'Memory Usage (MB)',
            'gpt_eval_score': 'GPT Evaluation Score'
        },
        height=600  # Make the plot taller
    )
    
    # Adjust the legend position and layout
    fig_memory_time.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            orientation="v"
        ),
        margin=dict(r=150)  # Add right margin to accommodate legend
    )
    
    st.plotly_chart(fig_memory_time, use_container_width=True)
    
    # Model Performance Analysis
    st.subheader("Model Performance Analysis")
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        # GPT Evaluation Score
        fig_gpt = px.bar(
            filtered_df,
            x='model_name',
            y='gpt_eval_score',
            color='variant',
            barmode='group',
            title='GPT Evaluation Score by Model and Variant'
        )
        st.plotly_chart(fig_gpt, use_container_width=True)
        
        # Memory Usage
        fig_memory = px.bar(
            filtered_df,
            x='model_name',
            y='memory_mb',
            color='variant',
            barmode='group',
            title='Memory Usage (MB) by Model and Variant'
        )
        st.plotly_chart(fig_memory, use_container_width=True)
    
    with perf_col2:
        # Syntax Evaluation Score
        fig_syntax = px.bar(
            filtered_df,
            x='model_name',
            y='syntax_eval_score',
            color='variant',
            barmode='group',
            title='Syntax Evaluation Score by Model and Variant'
        )
        st.plotly_chart(fig_syntax, use_container_width=True)
        
        # Wall Time
        fig_time = px.bar(
            filtered_df,
            x='model_name',
            y='wall_time_seconds',
            color='variant',
            barmode='group',
            title='Wall Time (seconds) by Model and Variant'
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    st.subheader("Performance vs Resource Trade-off")
    
    # Calculate overall score (50% GPT eval, 50% syntax eval)
    filtered_df['overall_score'] = (
        0.5 * filtered_df['gpt_eval_score'] + 
        0.5 * filtered_df['syntax_eval_score']
    )
    
    # Calculate overall resource usage
    filtered_df['resource_usage'] = (
        filtered_df['memory_mb'] / filtered_df['memory_mb'].max() + 
        filtered_df['wall_time_seconds'] / filtered_df['wall_time_seconds'].max()
    ) / 2
    
    fig_tradeoff = px.scatter(
        filtered_df,
        x='resource_usage',
        y='overall_score',
        color='model_name',
        symbol='variant',
        size='gpt_eval_score',
        hover_data=['syntax_eval_score', 'memory_mb', 'wall_time_seconds'],
        title='Performance vs Resource Usage Trade-off',
        labels={
            'resource_usage': 'Normalized Resource Usage (Memory + Time)',
            'overall_score': 'Overall Performance Score'
        },
        height=600
    )
    
    fig_tradeoff.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        margin=dict(r=150)
    )
    
    st.plotly_chart(fig_tradeoff, use_container_width=True)
    
    # Leaderboard section
    st.subheader("🏆 Model Leaderboard")
    
    # Create leaderboard dataframe with weighted scores
    leaderboard_df = filtered_df.copy()
    
    # Calculate weighted score (50% GPT eval, 50% syntax eval)
    leaderboard_df['overall_score'] = (
        0.5 * leaderboard_df['gpt_eval_score'] + 
        0.5 * leaderboard_df['syntax_eval_score']
    )
    
    # Sort by overall score
    leaderboard_df = leaderboard_df.sort_values('overall_score', ascending=False)
    
    # Format the display columns
    display_df = leaderboard_df[[
        'model_name', 
        'variant', 
        'overall_score',
        'gpt_eval_score',
        'syntax_eval_score',
        'memory_mb',
        'wall_time_seconds'
    ]].copy()
    
    # Round numeric columns
    display_df['overall_score'] = display_df['overall_score'].round(3)
    display_df['gpt_eval_score'] = display_df['gpt_eval_score'].round(3)
    display_df['syntax_eval_score'] = display_df['syntax_eval_score'].round(3)
    display_df['memory_mb'] = display_df['memory_mb'].round(1)
    display_df['wall_time_seconds'] = display_df['wall_time_seconds'].round(1)
    
    # Rename columns for better display
    display_df.columns = [
        'Model',
        'Variant',
        'Overall Score',
        'GPT Score',
        'Syntax Score',
        'Memory (MB)',
        'Time (s)'
    ]
    
    # Display the leaderboard with alternative styling
    st.dataframe(
        display_df.style
            .highlight_max(subset=['Overall Score', 'GPT Score', 'Syntax Score'], 
                         color='#70DD70')  # darker green
            .highlight_min(subset=['Memory (MB)', 'Time (s)'], 
                         color='#70DD70')  # darker green
            .highlight_min(subset=['Overall Score', 'GPT Score', 'Syntax Score'], 
                         color='#FF9999')  # darker red
            .highlight_max(subset=['Memory (MB)', 'Time (s)'], 
                         color='#FF9999'),  # darker red
        use_container_width=True,
        height=400,
        hide_index=True
    )

if __name__ == "__main__":
    main() 
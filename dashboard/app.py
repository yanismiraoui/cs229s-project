import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import glob

st.set_page_config(page_title="BOLT⚡️ Dashboard", layout="wide")

def load_results(results_dir="results"):
    """Load all JSON results files from the results directory."""
    results = []
    for file_path in glob.glob(f"{results_dir}/*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Add filename as run_id
            data['run_id'] = Path(file_path).stem
            results.append(data)
    return results

def create_results_df(results):
    """Convert results list to a pandas DataFrame with flattened structure."""
    flat_data = []
    for result in results:
        flat_result = {
            'run_id': result['run_id'],
            'model_name': result['model']['name'],
            'quantization_enabled': result['model']['configuration']['quantization']['enabled'],
            'quantization_bits': result['model']['configuration']['quantization']['bits'],
            'pruning_enabled': result['model']['configuration']['pruning']['enabled'],
            'pruning_target_sparsity': result['model']['configuration']['pruning']['target_sparsity'],
            'eval_fraction': result['data']['eval_fraction'],
            'num_test_examples': result['data']['num_test_examples'],
            'average_loss': result['metrics']['average_loss'],
            'average_similarity_loss': result['metrics']['average_similarity_loss'],
            'average_levenshtein_distance': result['metrics']['average_levenshtein_distance'],
            'wall_time_seconds': result['metrics']['wall_time_seconds'],
            'total_memory_mb': result['metrics']['total_memory_mb'],
            'total_flops_g': result['metrics']['total_flops_g'],
            'average_flops_per_sample_g': result['metrics']['average_flops_per_sample_g'],
            'device': result['runtime']['device'],
            'timestamp': result['runtime']['timestamp']
        }
        flat_data.append(flat_result)
    return pd.DataFrame(flat_data)

def main():
    st.title("BOLT⚡️ Dashboard")
    
    # Load results
    results = load_results()
    if not results:
        st.error("No results files found in the results directory!")
        return
    
    df = create_results_df(results)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=df['model_name'].unique(),
        default=df['model_name'].unique()
    )
    
    # Filter dataframe based on model selection first
    filtered_df = df[df['model_name'].isin(selected_models)]
    
    # Optimization filters
    st.sidebar.subheader("Optimization Filters")
    
    # Quantization filters
    quant_options = ['Quantized', 'Non-Quantized']
    selected_quant = st.sidebar.multiselect(
        "Quantization Status",
        options=quant_options,
        default=quant_options,
        help="Select models based on quantization status"
    )
    
    # Convert selection to boolean filter
    if selected_quant:
        quant_filter = filtered_df['quantization_enabled'].isin([q == 'Quantized' for q in selected_quant])
        filtered_df = filtered_df[quant_filter]
    
    # Show quantization bits filter only if we have quantized models in selection
    if 'Quantized' in selected_quant and not filtered_df.empty:
        available_bits = sorted([b for b in filtered_df['quantization_bits'].unique() if b is not None])
        if available_bits:
            selected_bits = st.sidebar.multiselect(
                "Quantization Bits",
                options=available_bits,
                default=available_bits,
                help="Select specific quantization bits"
            )
            if selected_bits:
                filtered_df = filtered_df[
                    (filtered_df['quantization_bits'].isin(selected_bits)) |
                    (filtered_df['quantization_bits'].isna())
                ]
    
    # Pruning filters
    pruning_options = ['Pruned', 'Non-Pruned']
    selected_pruning = st.sidebar.multiselect(
        "Pruning Status",
        options=pruning_options,
        default=pruning_options,
        help="Select models based on pruning status"
    )
    
    # Convert selection to boolean filter
    if selected_pruning:
        pruning_filter = filtered_df['pruning_enabled'].isin([p == 'Pruned' for p in selected_pruning])
        filtered_df = filtered_df[pruning_filter]
    
    # # Show sparsity range slider only if we have pruned models in selection
    # if 'Pruned' in selected_pruning and not filtered_df.empty:
    #     pruned_df = filtered_df[filtered_df['pruning_enabled'] == True]
    #     if not pruned_df.empty:
    #         min_sparsity = float(pruned_df['pruning_target_sparsity'].min())
    #         max_sparsity = float(pruned_df['pruning_target_sparsity'].max())
            
    #         # Handle case where min and max are equal
    #         if min_sparsity == max_sparsity:
    #             max_sparsity += 0.01  # Add a small buffer
    #             min_sparsity = max(0.0, min_sparsity - 0.01)  # Ensure we don't go below 0
            
    #         sparsity_ranges = st.sidebar.slider(
    #             "Pruning Sparsity Range",
    #             min_value=min_sparsity,
    #             max_value=max_sparsity,
    #             value=(min_sparsity, max_sparsity),
    #             step=0.01,
    #             help="Filter pruned models by sparsity range"
    #         )
    #         filtered_df = filtered_df[
    #             (filtered_df['pruning_enabled'] == False) |
    #             (
    #                 (filtered_df['pruning_target_sparsity'] >= sparsity_ranges[0]) &
    #                 (filtered_df['pruning_target_sparsity'] <= sparsity_ranges[1])
    #             )
    #         ]
    
    # Resource Usage Analysis
    st.subheader("Resource Usage Analysis")
    fig_memory_flops = px.scatter(
        filtered_df,
        x='total_flops_g',
        y='total_memory_mb',
        color='model_name',
        size='average_loss',
        hover_data=['timestamp', 'quantization_enabled', 'pruning_enabled'],
        title='Memory vs FLOPS Usage',
        labels={
            'total_flops_g': 'Total FLOPS (G)',
            'total_memory_mb': 'Total Memory (MB)',
            'average_loss': 'Average Loss'
        }
    )
    fig_memory_flops.update_traces(marker=dict(sizeref=2.*max(filtered_df['average_loss'])/(40.**2)))
    st.plotly_chart(fig_memory_flops, use_container_width=True)
    
    # Levenshtein Distance Analysis
    st.subheader("Levenshtein Distance Analysis")
    lev_col1, lev_col2 = st.columns(2)
    
    with lev_col1:
        # Levenshtein Distance by Model
        fig_levenshtein = px.bar(
            filtered_df,
            x='model_name',
            y='average_levenshtein_distance',
            title='Average Levenshtein Distance by Model',
            color='model_name'
        )
        fig_levenshtein.update_layout(showlegend=False)
        st.plotly_chart(fig_levenshtein, use_container_width=True)
    
    with lev_col2:
        # Levenshtein Distance vs FLOPS
        fig_lev_flops = px.scatter(
            filtered_df,
            x='total_flops_g',
            y='average_levenshtein_distance',
            color='model_name',
            size='total_memory_mb',
            hover_data=['quantization_enabled', 'pruning_enabled'],
            title='Levenshtein Distance vs FLOPS',
            labels={
                'total_flops_g': 'Total FLOPS (G)',
                'average_levenshtein_distance': 'Average Levenshtein Distance',
                'total_memory_mb': 'Memory Usage (MB)'
            }
        )
        fig_lev_flops.update_traces(marker=dict(sizeref=2.*max(filtered_df['total_memory_mb'])/(40.**2)))
        st.plotly_chart(fig_lev_flops, use_container_width=True)
    
    # Optimization Impact Section
    st.subheader("Optimization Configurations")
    opt_col1, opt_col2 = st.columns(2)
    
    with opt_col1:
        # Quantization Impact
        quant_fig = px.scatter(
            filtered_df,
            x='quantization_bits',
            y='average_loss',
            color='model_name',
            size='total_memory_mb',
            title='Quantization Impact on Loss',
            hover_data=['quantization_enabled']
        )
        st.plotly_chart(quant_fig, use_container_width=True)
    
    with opt_col2:
        # Pruning Impact
        pruning_fig = px.scatter(
            filtered_df,
            x='pruning_target_sparsity',
            y='average_loss',
            color='model_name',
            size='total_memory_mb',
            title='Pruning Impact on Loss',
            hover_data=['pruning_enabled']
        )
        st.plotly_chart(pruning_fig, use_container_width=True)
    
    # Performance Metrics (moved to bottom)
    st.subheader("Performance Metrics")
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        # Average Loss
        fig_avg_loss = px.bar(
            filtered_df,
            x='model_name',
            y='average_loss',
            title='Average Loss by Model',
            color='model_name'
        )
        fig_avg_loss.update_layout(showlegend=False)
        st.plotly_chart(fig_avg_loss, use_container_width=True)
        
        # Total Memory Usage
        fig_memory = px.bar(
            filtered_df,
            x='model_name',
            y='total_memory_mb',
            title='Total Memory Usage (MB)',
            color='model_name'
        )
        fig_memory.update_layout(showlegend=False)
        st.plotly_chart(fig_memory, use_container_width=True)
        
        # Wall Time
        fig_time = px.bar(
            filtered_df,
            x='model_name',
            y='wall_time_seconds',
            title='Execution Time (seconds)',
            color='model_name'
        )
        fig_time.update_layout(showlegend=False)
        st.plotly_chart(fig_time, use_container_width=True)
    
    with perf_col2:
        # Similarity Loss
        fig_sim_loss = px.bar(
            filtered_df,
            x='model_name',
            y='average_similarity_loss',
            title='Average Similarity Loss by Model',
            color='model_name'
        )
        fig_sim_loss.update_layout(showlegend=False)
        st.plotly_chart(fig_sim_loss, use_container_width=True)
        
        # Total FLOPS
        fig_flops = px.bar(
            filtered_df,
            x='model_name',
            y='total_flops_g',
            title='Total FLOPS (G)',
            color='model_name'
        )
        fig_flops.update_layout(showlegend=False)
        st.plotly_chart(fig_flops, use_container_width=True)
        
        # FLOPS per Sample
        fig_flops_sample = px.bar(
            filtered_df,
            x='model_name',
            y='average_flops_per_sample_g',
            title='Average FLOPS per Sample (G)',
            color='model_name'
        )
        fig_flops_sample.update_layout(showlegend=False)
        st.plotly_chart(fig_flops_sample, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    st.dataframe(
        filtered_df.style.highlight_max(axis=0, color='lightgreen')
                       .highlight_min(axis=0, color='lightpink'),
        use_container_width=True
    )

if __name__ == "__main__":
    main() 
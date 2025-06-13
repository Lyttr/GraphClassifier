import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community as community_louvain
import powerlaw
from sklearn.manifold import TSNE
from itertools import combinations
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

def analyze_centrality(G):
    """Analyze centrality metrics in the network"""
    # Initialize results dictionary
    centrality_results = {}
    
    # Basic centrality metrics
    centrality_results['degree'] = nx.degree_centrality(G)
    centrality_results['betweenness'] = nx.betweenness_centrality(G)
    centrality_results['eigenvector'] = nx.eigenvector_centrality(G)
    centrality_results['closeness'] = nx.closeness_centrality(G)
    centrality_results['harmonic'] = nx.harmonic_centrality(G)
    
    # Get top 5 most central nodes
    top_deg = sorted(centrality_results['degree'].items(), key=lambda x: x[1], reverse=True)[:5]
    top_btw = sorted(centrality_results['betweenness'].items(), key=lambda x: x[1], reverse=True)[:5]
    top_eig = sorted(centrality_results['eigenvector'].items(), key=lambda x: x[1], reverse=True)[:5]
    top_close = sorted(centrality_results['closeness'].items(), key=lambda x: x[1], reverse=True)[:5]
    top_harmonic = sorted(centrality_results['harmonic'].items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("Top 5 nodes by degree centrality:", top_deg)
    print("Top 5 nodes by betweenness centrality:", top_btw)
    print("Top 5 nodes by eigenvector centrality:", top_eig)
    print("Top 5 nodes by closeness centrality:", top_close)
    print("Top 5 nodes by harmonic centrality:", top_harmonic)
    
    return centrality_results

def analyze_top_percent_overlap(centrality_results, percent=0.1):
    """Analyze overlap of top percent% nodes between different centrality metrics"""
    def top_percent_nodes(centrality_dict, percent=0.1):
        N = int(len(centrality_dict) * percent)
        return set(sorted(centrality_dict, key=centrality_dict.get, reverse=True)[:N])
    
    # Get top percent% nodes for each centrality metric
    top_nodes = {}
    for name, centrality_dict in centrality_results.items():
        top_nodes[name] = top_percent_nodes(centrality_dict, percent)
        print(f"Top {percent*100}% nodes by {name}: {len(top_nodes[name])} nodes")
    
    # Calculate overlap between all centrality metrics
    overlaps = {}
    for i, (name1, nodes1) in enumerate(top_nodes.items()):
        for name2, nodes2 in list(top_nodes.items())[i+1:]:
            overlap = nodes1 & nodes2
            overlaps[f"{name1}-{name2}"] = overlap
            print(f"Overlap between {name1} and {name2}: {len(overlap)} nodes")
    
    return overlaps

def detect_communities(G):
    """Detect communities using Louvain algorithm"""
    communities = community_louvain.best_partition(G)
    
    # Count community sizes
    community_sizes = {}
    for node, community in communities.items():
        if community not in community_sizes:
            community_sizes[community] = 0
        community_sizes[community] += 1
    
    print("Community sizes:", sorted(community_sizes.items(), key=lambda x: x[1], reverse=True))
    
    return communities

def analyze_degree_distribution(G):
    """Analyze degree distribution"""
    degrees = [d for n, d in G.degree()]
    
    # Fit using powerlaw library
    fit = powerlaw.Fit(degrees)
    
    # Plot degree distribution
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50, density=True, alpha=0.7)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()
    
    print("Power law exponent:", fit.power_law.alpha)
    print("Power law xmin:", fit.power_law.xmin)
    print("Power law sigma:", fit.power_law.sigma)
    
    # Perform goodness-of-fit test
    R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    print(f"Power law vs Exponential comparison: R={R}, p={p}")
    
    return degrees

def analyze_network_metrics(G):
    """Analyze basic network metrics"""
    diameter = nx.diameter(G)
    avg_shortest_path = nx.average_shortest_path_length(G)
    
    print(f"Network diameter: {diameter}")
    print(f"Average shortest path length: {avg_shortest_path}")
    
    return diameter, avg_shortest_path

def compare_with_model_graphs(G):
    """Compare with model graphs and provide quantitative similarity measures"""
    n = len(G)
    m = G.number_of_edges()
    p = 2 * m / (n * (n-1))  # Average degree
    
    print(f"Original graph statistics:")
    print(f"Nodes: {n}, Edges: {m}, Average degree: {p:.4f}")
    
    # Generate model graphs
    ER = nx.erdos_renyi_graph(n, p)
    WS = nx.watts_strogatz_graph(n, int(p*n/2), 0.1)
    BA = nx.barabasi_albert_graph(n, int(p*n/2))
    
    # Get degree sequences
    original_degrees = np.array([d for n, d in G.degree()])
    er_degrees = np.array([d for n, d in ER.degree()])
    ws_degrees = np.array([d for n, d in WS.degree()])
    ba_degrees = np.array([d for n, d in BA.degree()])
    
    # Calculate quantitative similarity measures
    print(f"\n=== Quantitative Degree Distribution Similarity Analysis ===")
    
    models = {
        'Erdős-Rényi (ER)': er_degrees,
        'Watts-Strogatz (WS)': ws_degrees,
        'Barabási-Albert (BA)': ba_degrees
    }
    
    similarity_results = {}
    
    for model_name, model_degrees in models.items():
        print(f"\n{model_name} Model Similarity Analysis:")
        
        # 1. Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(original_degrees, model_degrees)
        print(f"  KS Test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
        
        # 2. Wasserstein distance (Earth Mover's Distance)
        wasserstein_dist = wasserstein_distance(original_degrees, model_degrees)
        print(f"  Wasserstein Distance: {wasserstein_dist:.4f}")
        
        # 3. Create normalized histograms for JS divergence
        max_degree = max(max(original_degrees), max(model_degrees))
        bins = np.arange(0, max_degree + 2) - 0.5
        
        original_hist, _ = np.histogram(original_degrees, bins=bins, density=True)
        model_hist, _ = np.histogram(model_degrees, bins=bins, density=True)
        
        # Normalize to create probability distributions
        original_prob = original_hist / np.sum(original_hist)
        model_prob = model_hist / np.sum(model_hist)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        original_prob = original_prob + epsilon
        model_prob = model_prob + epsilon
        original_prob = original_prob / np.sum(original_prob)
        model_prob = model_prob / np.sum(model_prob)
        
        # 4. Jensen-Shannon divergence
        js_divergence = jensenshannon(original_prob, model_prob)
        print(f"  Jensen-Shannon Divergence: {js_divergence:.4f}")
        
        # 5. Basic statistical comparisons
        original_mean = np.mean(original_degrees)
        model_mean = np.mean(model_degrees)
        original_std = np.std(original_degrees)
        model_std = np.std(model_degrees)
        original_skew = stats.skew(original_degrees)
        model_skew = stats.skew(model_degrees)
        
        print(f"  Statistical Measures Comparison:")
        print(f"    Mean degree: Original={original_mean:.4f}, Model={model_mean:.4f}, Difference={abs(original_mean-model_mean):.4f}")
        print(f"    Standard deviation: Original={original_std:.4f}, Model={model_std:.4f}, Difference={abs(original_std-model_std):.4f}")
        print(f"    Skewness: Original={original_skew:.4f}, Model={model_skew:.4f}, Difference={abs(original_skew-model_skew):.4f}")
        
        # 6. Correlation coefficient between sorted degree sequences
        original_sorted = np.sort(original_degrees)
        model_sorted = np.sort(model_degrees)
        correlation, corr_pvalue = stats.pearsonr(original_sorted, model_sorted)
        print(f"  Sorted degree sequence correlation: r={correlation:.4f}, p-value={corr_pvalue:.4f}")
        
        # Store results
        similarity_results[model_name] = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'wasserstein_distance': wasserstein_dist,
            'js_divergence': js_divergence,
            'mean_difference': abs(original_mean - model_mean),
            'std_difference': abs(original_std - model_std),
            'skew_difference': abs(original_skew - model_skew),
            'correlation': correlation,
            'correlation_pvalue': corr_pvalue
        }
    
    # Find the most similar model
    print(f"\n=== Comprehensive Similarity Ranking ===")
    print("Multi-metric comprehensive evaluation:")
    
    # Calculate composite similarity scores (lower is better for most metrics)
    composite_scores = {}
    for model_name, results in similarity_results.items():
        # Normalize and combine scores (inverse for correlation since higher is better)
        score = (results['ks_statistic'] + 
                results['wasserstein_distance']/100 +  # Scale down Wasserstein
                results['js_divergence'] + 
                results['mean_difference'] + 
                results['std_difference']/10 +  # Scale down std difference
                results['skew_difference'] + 
                (1 - results['correlation']))  # Inverse correlation
        composite_scores[model_name] = score
        print(f"{model_name}: Composite Score = {score:.4f}")
    
    # Sort by similarity (lower score is better)
    ranked_models = sorted(composite_scores.items(), key=lambda x: x[1])
    print(f"\nMost similar model: {ranked_models[0][0]}")
    
    # Compare degree distributions with enhanced visualization
    plt.figure(figsize=(20, 12))
    
    # Subplot 1: Original vs ER
    plt.subplot(2, 3, 1)
    plt.hist(original_degrees, bins=50, alpha=0.7, label='Original', density=True)
    plt.hist(er_degrees, bins=50, alpha=0.7, label='ER Model', density=True)
    plt.title(f"Degree Distribution Comparison: Original vs ER\nKS Statistic: {similarity_results['Erdős-Rényi (ER)']['ks_statistic']:.4f}")
    plt.xlabel("Degree")
    plt.ylabel("Probability Density")
    plt.legend()
    
    # Subplot 2: Original vs WS
    plt.subplot(2, 3, 2)
    plt.hist(original_degrees, bins=50, alpha=0.7, label='Original', density=True)
    plt.hist(ws_degrees, bins=50, alpha=0.7, label='WS Model', density=True)
    plt.title(f"Degree Distribution Comparison: Original vs WS\nKS Statistic: {similarity_results['Watts-Strogatz (WS)']['ks_statistic']:.4f}")
    plt.xlabel("Degree")
    plt.ylabel("Probability Density")
    plt.legend()
    
    # Subplot 3: Original vs BA
    plt.subplot(2, 3, 3)
    plt.hist(original_degrees, bins=50, alpha=0.7, label='Original', density=True)
    plt.hist(ba_degrees, bins=50, alpha=0.7, label='BA Model', density=True)
    plt.title(f"Degree Distribution Comparison: Original vs BA\nKS Statistic: {similarity_results['Barabási-Albert (BA)']['ks_statistic']:.4f}")
    plt.xlabel("Degree")
    plt.ylabel("Probability Density")
    plt.legend()
    
    # Subplot 4: Log-log scale comparison
    plt.subplot(2, 3, 4)
    degrees_unique, degrees_count = np.unique(original_degrees, return_counts=True)
    plt.loglog(degrees_unique, degrees_count/len(original_degrees), 'o-', label='Original', alpha=0.7)
    
    er_unique, er_count = np.unique(er_degrees, return_counts=True)
    plt.loglog(er_unique, er_count/len(er_degrees), 's-', label='ER Model', alpha=0.7)
    
    plt.title("Degree Distribution (Log-Log Scale)")
    plt.xlabel("Degree (log)")
    plt.ylabel("Probability (log)")
    plt.legend()
    
    # Subplot 5: Cumulative distribution comparison
    plt.subplot(2, 3, 5)
    plt.hist(original_degrees, bins=50, alpha=0.7, cumulative=True, density=True, label='Original')
    plt.hist(er_degrees, bins=50, alpha=0.7, cumulative=True, density=True, label='ER Model')
    plt.hist(ws_degrees, bins=50, alpha=0.7, cumulative=True, density=True, label='WS Model')
    plt.hist(ba_degrees, bins=50, alpha=0.7, cumulative=True, density=True, label='BA Model')
    plt.title("Cumulative Distribution Function Comparison")
    plt.xlabel("Degree")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    
    # Subplot 6: Q-Q plot for best matching model
    best_model_name = ranked_models[0][0]
    best_model_degrees = models[best_model_name]
    plt.subplot(2, 3, 6)
    
    # Manual Q-Q plot creation
    # Sort both degree sequences
    original_sorted = np.sort(original_degrees)
    model_sorted = np.sort(best_model_degrees)
    
    # Create quantiles for comparison
    n_original = len(original_sorted)
    n_model = len(model_sorted)
    
    # Use the smaller sample size to determine number of quantiles
    n_quantiles = min(n_original, n_model)
    quantile_positions = np.linspace(0, 1, n_quantiles)
    
    # Calculate quantiles for both distributions
    original_quantiles = np.quantile(original_degrees, quantile_positions)
    model_quantiles = np.quantile(best_model_degrees, quantile_positions)
    
    # Plot Q-Q plot
    plt.scatter(model_quantiles, original_quantiles, alpha=0.6, s=20)
    
    # Add diagonal reference line (perfect match)
    min_val = min(np.min(model_quantiles), np.min(original_quantiles))
    max_val = max(np.max(model_quantiles), np.max(original_quantiles))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Match')
    
    plt.title(f"Q-Q Plot: Original vs {best_model_name}")
    plt.xlabel(f"{best_model_name} Theoretical Quantiles")
    plt.ylabel("Original Sample Quantiles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ER, WS, BA, similarity_results

def visualize_network_with_overlaps(G, overlaps, centrality_results, top_n=5):
    """Create separate visualizations for each pair of centrality metrics' overlaps, including t-SNE and spring layouts"""
    # Get all centrality metric pairs
    centrality_pairs = list(combinations(centrality_results.keys(), 2))
    
    for pair in centrality_pairs:
        metric1, metric2 = pair
        print(f"\nAnalyzing overlapping nodes between {metric1} and {metric2}...")
        
        # Get top N nodes for each metric pair
        top_nodes1 = set(sorted(centrality_results[metric1].items(), key=lambda x: x[1], reverse=True)[:top_n])
        top_nodes2 = set(sorted(centrality_results[metric2].items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        # Find overlapping nodes
        nodes1_set = {node for node, _ in top_nodes1}
        nodes2_set = {node for node, _ in top_nodes2}
        overlapping_nodes = nodes1_set & nodes2_set
        
        if not overlapping_nodes:
            print(f"No overlapping nodes found between {metric1} and {metric2}")
            continue
            
        # Create subgraph containing only overlapping nodes and their neighbors
        subgraph_nodes = set(overlapping_nodes)
        for node in overlapping_nodes:
            subgraph_nodes.update(G.neighbors(node))
        subgraph = G.subgraph(subgraph_nodes)
        
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. Spring layout visualization
        pos_spring = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Set node colors
        node_colors = []
        for node in subgraph.nodes():
            if node in overlapping_nodes:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        # Set node sizes
        node_sizes = [subgraph.degree(node) * 10 for node in subgraph.nodes()]
        
        # Draw spring layout
        nx.draw_networkx_nodes(subgraph, pos_spring, 
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.7,
                             ax=ax1)
        nx.draw_networkx_edges(subgraph, pos_spring, alpha=0.2, ax=ax1)
        
        # Add labels only for overlapping nodes
        labels = {node: node for node in overlapping_nodes}
        nx.draw_networkx_labels(subgraph, pos_spring, labels, font_size=8, ax=ax1)
        
        ax1.set_title(f'Spring Layout: {metric1} vs {metric2} Overlap')
        ax1.axis('off')
        
        # 2. t-SNE visualization
        try:
            # Create node feature matrix
            node_features = []
            node_list = list(subgraph.nodes())
            for node in node_list:
                features = []
                for metric in centrality_results.values():
                    features.append(metric.get(node, 0))
                node_features.append(features)
            
            # Convert feature list to numpy array
            X = np.array(node_features)
            
            # Skip t-SNE if too few nodes
            if len(X) < 5:
                print(f"Too few nodes ({len(X)}), skipping t-SNE visualization")
                ax2.text(0.5, 0.5, f"Too few nodes ({len(X)}),\nCannot perform t-SNE visualization",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                ax2.axis('off')
            else:
                # Apply t-SNE
                perplexity = min(30, len(X)-1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                node_positions = tsne.fit_transform(X)
                
                # Create node position dictionary
                pos_tsne = {node: pos for node, pos in zip(node_list, node_positions)}
                
                # Draw t-SNE layout
                nx.draw_networkx_nodes(subgraph, pos_tsne,
                                     node_color=node_colors,
                                     node_size=node_sizes,
                                     alpha=0.7,
                                     ax=ax2)
                nx.draw_networkx_edges(subgraph, pos_tsne, alpha=0.2, ax=ax2)
                nx.draw_networkx_labels(subgraph, pos_tsne, labels, font_size=8, ax=ax2)
                
                ax2.set_title(f't-SNE Layout: {metric1} vs {metric2} Overlap')
                ax2.axis('off')
        except Exception as e:
            print(f"t-SNE visualization failed: {str(e)}")
            ax2.text(0.5, 0.5, f"t-SNE visualization failed:\n{str(e)}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
            ax2.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='red', markersize=10, label='Overlapping Nodes'),
            Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='lightblue', markersize=10, label='Other Nodes')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed information about overlapping nodes
        print(f"\nOverlapping nodes between {metric1} and {metric2}:")
        for node in overlapping_nodes:
            print(f"\nNode {node}:")
            print(f"  {metric1}: {centrality_results[metric1][node]:.4f}")
            print(f"  {metric2}: {centrality_results[metric2][node]:.4f}")

def main():
    # Read Facebook network data
    G = nx.read_edgelist('dataset/facebook_combined.txt')
    
    print("=== Centrality Analysis ===")
    # centrality_results = analyze_centrality(G)
    
    # print("\n=== Top 10% Overlap Analysis ===")
    # overlaps = analyze_top_percent_overlap(centrality_results)
    
    # # print("\n=== Visualizing Overlaps ===")
    # # visualize_overlaps(overlaps, centrality_results)
    
    # print("\n=== Visualizing Network with Overlapping Nodes ===")
    # visualize_network_with_overlaps(G, overlaps, centrality_results)
    
    # print("\n=== Community Detection ===")
    # communities = detect_communities(G)
    
    # print("\n=== Degree Distribution Analysis ===")
    # degrees = analyze_degree_distribution(G)
    
    # print("\n=== Network Metrics ===")
    # diameter, avg_shortest_path = analyze_network_metrics(G)
    
    print("\n=== Model Graph Comparison ===")
    ER, WS, BA, similarity_results = compare_with_model_graphs(G)

if __name__ == "__main__":
    main() 
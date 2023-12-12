def cos_sim(a, b):
  
    if not isinstance(a, tf.Tensor):
        a = tf.convert_to_tensor(a)

    if not isinstance(b, tf.Tensor):
        b = tf.convert_to_tensor(b)

    if len(a.shape) == 1:
        a = tf.expand_dims(a, 0)

    if len(b.shape) == 1:
        b = tf.expand_dims(b, 0)

    a_norm = tf.nn.l2_normalize(a, axis=1)
    b_norm = tf.nn.l2_normalize(b, axis=1)
    return tf.matmul(a_norm, b_norm, transpose_b=True)

def community_detection(embeddings, threshold=0.75, min_community_size=10, batch_size=1024):

    threshold = tf.convert_to_tensor(threshold, dtype=embeddings.dtype)

    extracted_communities = []

    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    num_batches = (len(embeddings) - 1) // batch_size + 1

    for start_idx in tqdm(range(0, len(embeddings), batch_size), total=num_batches, desc='Processing batches'):

        cos_scores = cos_sim(embeddings[start_idx:start_idx + batch_size], embeddings)

        top_k_values, _ = tf.nn.top_k(cos_scores, k=min_community_size)

        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                new_cluster = []

                top_val_large, top_idx_large = tf.nn.top_k(cos_scores[i], k=sort_max_size)

                while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                    sort_max_size = min(2 * sort_max_size, len(embeddings))
                    top_val_large, top_idx_large = tf.nn.top_k(cos_scores[i], k=sort_max_size)

                for idx, val in zip(top_idx_large.numpy(), top_val_large.numpy()):
                    if val < threshold:
                        break

                    new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        del cos_scores

    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        community = sorted(community)
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities

def recall_and_filtration(
    our_hits, hmmer_hits, distance_threshold, comp_func, evalue_threshold
):
    match_count = 0
    our_total_hits = 0
    hmmer_hits_for_our_queries = 0
    # since we sometimes don't have
    # all queries, iterate over the DB in this fashion.
    for query in our_hits:
        if query not in hmmer_hits:
            # we've set an e-value threshold, meaning
            # that this query was never picked up
            print(f"Query {query} not in hmmer_hits.")

        matches = hmmer_hits[query]
        filtered = {}
        for match, evalue in matches.items():
            evalue = float(evalue[0])
            if evalue <= evalue_threshold:
                filtered[match] = evalue

        true_matches = filtered
        hmmer_hits_for_our_queries += len(true_matches)
        our_matches = our_hits[query]
        for match in our_matches:
            if comp_func(our_matches[match], distance_threshold):
                if match in true_matches:
                    # count the matches for each query.
                    # pdb.set_trace()
                    match_count += 1
                our_total_hits += 1

    # total hmmer hits
    denom = hmmer_hits_for_our_queries
    return 100 * (match_count / denom), our_total_hits

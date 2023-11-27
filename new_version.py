def calculate_top_1000_user(utility_mx):
    n_ratings_df = pd.DataFrame(index=utility_mx.index, columns=["n_ratings"])
    
    for index, row in n_ratings_df.iterrows():
        row["n_ratings"] = len(utility_mx.loc[index,:].dropna())
    
    n_ratings_df = n_ratings_df.sort_values("n_ratings", ascending=False)
    np.savetxt("top_1000_user.csv", n_ratings_df.iloc[:1000].index, delimiter=",", header="user_id")
    return n_ratings_df.iloc[:1000].index

def similarUsers(userID, nCommonMovies):
    #Assembling utility matrix, where row represent users (by User_ID), columns are movies (by Movie_ID), and the values are the ratings given to that movie by that user
    uti_mx = df_ratings.pivot_table(values='Rating', index='User_ID', columns='Movie_ID', aggfunc='first')

    # Calculating a list of top 1000 user_ids with the most ratings
    top_1000 = []
    try:
        top_1000 = np.genfromtxt('top_1000_user.csv', delimiter=',')
        if len(top_1000) == 0:
            top_1000 = calculate_top_1000_user(uti_mx)
    except Exception as e:
        print("Problem occured while loading top users, calculating it live now...")
        top_1000 = calculate_top_1000_user(uti_mx)
    
    uti_mx_top_1000 = uti_mx.loc[top_1000]

    #Normalize utility matrix, subtracting mean of ratings from the actual ratings for each respective user
    norm_uti_mx = uti_mx_top_1000.sub(uti_mx_top_1000.mean(axis=1), axis=0)
    
    #Calculating Cosine distance of a highlighted user and every other user if they have at least 'n_common_movies' that they both rated
    #Number of common movies needed to be considering other user
    
    #init matrix
    cosine_dist_mx = pd.DataFrame(index=[userID], columns=norm_uti_mx.index)
    cosine_dist_mx = cosine_dist_mx.drop(userID, axis=1)
    
    chosen_set = set(norm_uti_mx.loc[userID,:].dropna().index)
    
    for user in top_1000:
        # list of common movie IDs
        common_list = list(chosen_set & set(norm_uti_mx.loc[user,:].dropna().index))
        if len(common_list) > nCommonMovies:
            a = norm_uti_mx.loc[userID, common_list].values
            b = norm_uti_mx.loc[user, common_list].values
            cosine_dist_mx[user] = distance.cosine(a,b)
            
    # sort by distance, lower distance means more similar
    cosine_dist_mx_sorted = cosine_dist_mx.sort_values(userID, axis=1).dropna(axis=1)
    similarity = cosine_dist_mx_sorted.T
    

    mostSimilarUsers = similarity[similarity[userID] == 0][userID].keys().to_list() #find users with similarity on 0 
    if not mostSimilarUsers: #if no users has similarity on 0
        mostSimilarUsers = similarity[userID].nsmallest(10).keys().to_list() #userID for 10 most similar users
    
    return mostSimilarUsers

userID = 305344
userSimilarity = similarUsers(userID, 20)
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self) -> None:
        self.team_colours = {}
        self.kmeans = None
        self.player_team_dict = {} # player_id => team_id

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)

        # kmeans++ helps to get better clusters faster
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        top_half_image = image[0: int(image.shape[0]/2), :]

        # clustering models
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels
        labels = kmeans.labels_

        # reshape the labels to the image shape
        clustered_image = labels.reshape(int(top_half_image.shape[0]), int(top_half_image.shape[1]))

        # get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_colour = kmeans.cluster_centers_[player_cluster]

        return player_colour

    def assign_team_colour(self, frame, player_detections):
        player_colours = []
        for _, player in player_detections.items():
            bbox = player['bbox']
            player_colour = self.get_player_color(frame, bbox)
            player_colours.append(player_colour)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(player_colours)

        self.kmeans = kmeans
        
        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)    

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id == 91: 
            team_id = 1

        self.player_team_dict[player_id] = team_id
        
        return team_id
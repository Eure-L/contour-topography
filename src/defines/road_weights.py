from data_models.road_weight import RoadWeight


class RoadsWeight:
    """ Ranking to interpolate roads thikness based on their hierarchy """

    CONSTANT = RoadWeight(
        [(0.4, 0x1),
         (0.4, 0x8a),
         (0.4, 0)
         ]
    )
    RANKING_1 = RoadWeight(
        [(0.6, 0x1),
         (0.4, 0x8a),
         (0.2, 0x8b)
         ]
    )

    RANKING_2 = RoadWeight(
        [(1.0, 0x1),
         (0.6, 0x8a),
         (0.2, 0x8b),
         (0.1, 0x9)]
    )

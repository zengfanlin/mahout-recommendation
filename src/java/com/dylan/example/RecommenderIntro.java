package com.dylan.example;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.*;
import org.apache.mahout.cf.taste.impl.similarity.*;
import org.apache.mahout.cf.taste.impl.neighborhood.*;
import org.apache.mahout.cf.taste.impl.recommender.*;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.*;
import org.apache.mahout.cf.taste.neighborhood.*;
import org.apache.mahout.cf.taste.recommender.*;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class RecommenderIntro {
    public static void main(String[] args) throws IOException, TasteException {
        DataModel model = new FileDataModel(new File("D:\\00-workspace\\01_java\\mahout-recommendation\\src\\java\\intro.csv"));
        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(3, similarity, model);//3个临近的用户
        Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
        List<RecommendedItem> itemList = recommender.recommend(2, 2);//给userid=1的用户推荐2个列表
        for (RecommendedItem item : itemList) {
            System.out.println(item);
//            RecommendedItem[item:109, value:5.0]
//            RecommendedItem[item:103, value:5.0]
        }
    }
}

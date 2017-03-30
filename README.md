# Bigdata_analysis_HW1
HW1 to find important feature

B10315011 NTUST 四資工三甲 林航平
執行結果有截圖在screenshot,(RF_classifier_top10_feature, RF_regressor_top10_feature)
Use sklearn RandonForestClassifier, RandomForestRegressor to find feature importances
Regressor假設class 1~9為實數去接近該數,可能可以看出class 1~9是否有和歸屬類別的數值1~9有特別意義
[Reference](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

1. important feature: 
 * ## RF_classifier
 * #Feature ranking (Feature importance)
 * 1. TB_9a	
 * 2. TB_a9	
 * 3. Img0.1
 * 4. ent_p_5	
 * 5. TB_b1
 * 6. TB_71	
 * 7. ent_p_8	
 * 8. TB_ce	
 * 9.GetStringTypeA	
 * 10. ExitProcess

 * ## RF_regressor
 * 1. __getmainargs	
 * 2. ent_p_diffs_5
 * 3. db3_NdNt	
 * 4. dc_por	
 * 5. section_names_header	
 * 6. ent_p_diffs_8	
 * 7. Offset.1	
 * 8. _initterm	
 * 9. Unknown_Sections_lines_por	
 * 10. char

2. Useless feature:

3. Use sklearn library's RandomForestRegressor in Python

4. sklearn library , (Using pandas to draw plot and check important feature)

5. 尚無建議

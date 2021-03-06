# Bigdata_analysis_HW1
HW1 to find important feature

B10315011 NTUST 四資工三甲 林航平
執行結果有截圖在screenshot,(RF_classifier_top10_feature, RF_regressor_top10_feature)
Use sklearn RandonForestClassifier, RandomForestRegressor to find feature importances
Regressor假設 class1到9 為實數去接近該數,可能可以看出 class1到9 是否有和歸屬類別的數值1~9有特別意義
[Reference](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

1. important feature: 
 * ## RF_classifier
 * #Feature ranking top10 (By Feature importance)
 * 1. TB_9a	
 * 2. TB_a9	
 * 3. Img0.1
 * 4. ent_p_5	
 * 5. TB_b1
 * 6. TB_71	
 * 7. ent_p_8	
 * 8. TB_ce	
 * 9. GetStringTypeA	
 * 10. ExitProcess

 * ## RF_regressor
 * #Feature ranking top10 (By Feature importance)
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

 Use those features and test in weka using LibSVM training set(the result are also in file Screenshot)
 Result is that we could classify **100%** correct with **Top 3 feature (TB_9a	TB_a9	Img0.1) generated by RF_classifier**
 
 In fact, we only need No.2 and No.3 (TB_a9	Img0.1) could achieve 100%
 
 However, if we use All Top 10 feature from RF_regressor, we could just classify 94% correct, so the classes may not closely related with their class number 1to9 
 
2. Useless feature:
 There are 10 features on the final list of importances(Top10 useless feature): 
 + GetLastActivePopup	
 + ImageList_Add	
 + GlobalDeleteAtom	
 + IsBadReadPtr	
 + SelectPalette	
 + GetMenuState	
 + ExitThread	
 + AdjustWindowRectEx 
 + GetEnvironmentVariableA	
 + SHGetFileInfoA
 
 Test these 10 features with weka using LibSVM training set, it is only 59% correct
 
3. Use sklearn library's RandonforestClassifier,取前10後10當important and useless features.
   結果也相當不錯，用兩個重要feature即可100%分類成功，而用importance分數最低的10個features則分類59%成功，但無法保證說這些features對分類**完全無效**，但**相較非常無效**
   
4. sklearn library 做feature selection, 用pandas,numpy處理資料, matplotlib.pyplot畫圖

5. 尚無建議

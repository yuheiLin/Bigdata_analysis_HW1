# Bigdata_analysis_HW1
HW1 to find important feature

B10315011 NTUST 四資工三甲 林航平

Use sklearn RandomForestRegressor to find feature importances

Reference :  [an example](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html "Title")

1. important feature: 
 # Feature ranking (Feature importance)
 * 1. feature 826 (0.397582)
 * 2. feature 282 (0.175932)
 * 3. feature 1513 (0.060665)
 * 4. feature 306 (0.038663)
 * 5. feature 1497 (0.036800)
 * 6. feature 1493 (0.023518)
 * 7. feature 285 (0.020039)
 * 8. feature 561 (0.018057)
 * 9. feature 25 (0.018032)
 * 10. feature 1430 (0.015106)

2. Useless feature:
  + After rank 13 ,feature 313 (0.010672),
  the rest of the importances are less than 1%, so those might be useless

3. Use sklearn library's RandomForestRegressor in Python

4. 

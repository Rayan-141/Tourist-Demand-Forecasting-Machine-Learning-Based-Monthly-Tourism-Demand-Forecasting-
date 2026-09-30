# Viva Questions — Tourist Demand Forecasting

## Project
1. What is your project?
2. What is the problem statement?
3. Why is this a machine-learning problem?
4. Who can use the system?
5. What is the input?
6. What is the output?

## Dataset
7. What dataset did you use?
8. Who is the data agency?
9. What is the original source?
10. What does one row in the original CSV represent?
11. Why did you convert the dataset from wide to long format?
12. What is the target variable?
13. What data-quality checks did you perform?

## ML
14. Is this supervised or unsupervised learning?
15. Why regression?
16. Why is it also called time-series forecasting?
17. Why do you not randomly shuffle the data?
18. What is data leakage?
19. What is a lag feature?
20. Why did you use Lag 12?
21. What is a rolling mean?
22. Why use month sine/cosine features?

## Preprocessing
23. What is missing-value handling?
24. Why not replace every missing value with zero?
25. What is one-hot encoding?
26. Why encode Month and Quarter?
27. What is feature scaling?
28. Why use StandardScaler?
29. Does Random Forest require scaling?
30. Why is preprocessing important?

## Regression
31. What is Linear Regression?
32. What is Polynomial Regression?
33. Why compare degree 2 and degree 4?
34. What is overfitting?
35. What is underfitting?

## Random Forest
36. What is Random Forest?
37. What is an ensemble?
38. How does Random Forest combine trees?
39. Why use Random Forest?
40. What is feature importance?

## Evaluation
41. What is MAE?
42. What is MSE?
43. What is RMSE?
44. What is R²?
45. Why use more than one metric?
46. What is cross-validation?
47. Why use TimeSeriesSplit instead of random K-fold?
48. What does a large error mean?
49. Why might COVID-period observations have larger errors?

## Streamlit
50. What is Streamlit?
51. How does app.py get the trained model?
52. Why did you not use a .pkl file?
53. What is st.cache_resource?
54. How does a future forecast get generated?
55. What happens when the user selects 12 months?
56. What is recursive forecasting?

## Limitations
57. Can your model predict a pandemic?
58. What external factors are missing?
59. Is a prediction guaranteed?
60. What would you improve in a future version?

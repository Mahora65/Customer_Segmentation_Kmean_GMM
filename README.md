# Case Study
**Problem type:** Customer Segmentation <br>

## General Context

This dataset contain sales data, produced by customer visiting and purchasing small items from various sources on daily basis. Therefore, in order to observe the purchasing behaviour and to identify the customer segmentation the following questions needed to be investigated: 

1. Describe the datasets and the eventual anomalies you find.
2. Which patterns do you find in the purchasing behavior of the customers?
3. What are the categories and products the customers are most interested in?
4. Split customers into different groups based on their purchasing behavior.<br>
   - Justify your choice for your adopted method(s) and model(s).<br>
   - Describe the defined customer groups. What are the features which are driving the differentiation amongst the different groups?
   - Give suggestions on how the business should treat these clusters differently.
5. Implement further ideas (initiatives, further analyses) you might have in mind which can be helpful for the business.<br>

## Metadata

Each unit (department, customer, product, etc.) has an associated unique id. Files and names should be self explanatory.

#### departments.csv

Contains a generalized grouping of the products available in the dataset.

```
department_id,department  
1,frozen  
2,other  
3,bakery  
...
```

#### order_products.csv

Contains previous order contents for all customers. 'reordered' indicates that the customer has a previous order that contains the product. Note that some orders will have no reordered items.

```
order_id,product_id,add_to_cart_order,reordered  
1934,83,1,1  
1934,37,2,1  
1934,66,3,0  
... 
```

#### orders.csv

Contains information for each order. 'order_dow', for example, represents the weekday.

```
order_id,user_id,order_number,order_dow,order_hour_of_day,days_since_prior_order  
2539329,1,1,2,08,  
2398795,1,2,3,07,15.0  
473747,1,3,3,12,21.0  
...
```

#### products.csv

Contains product information. 

```
product_id,department_id,product_name
1,19,Chocolate Sandwich Cookies  
2,13,All-Seasons Salt  
3,7,Robust Golden Unsweetened Oolong Tea  
...
```


**Have fun!**

Total number of rows 1,3719,80.



|name|type| description                                                                                                   |
|---|---|---------------------------------------------------------------------------------------------------------------|
|customer_id|string| uniq id                                                                                                       |
|FN|float| phone number indicator? 1.0 for 35% of customers                                                              |
|Active|float| some activity indicator? 1.0 for 34% of customers                                                             |
|club_member_status|string| h&m club membership 92.7% active, 6.7% pre-create, uknown .5% .03% left                                       |
|fashion_news_frequency|string| 2/3 None 1/3 Regularly .06 monthly                                                                            |
|age|int| customer's age in range [16-99] median age 32, iqr 24-49              a few missing 16k (1.15%)               |
|postal_code|string| 352,899 uniq values hashed postal codes, with special value for NA? iqr 1-5 customers on the same postal_code |

120,303 customers don't share the same postal_code: 2c29ae653a9282cce4151bd87643c907644e09541abc28ae87dea0d1f6603b1c, most probably N/A placeholder 

What is the relationship between `Active` and `club_member_status:Action`?

Most probably US customers, peaks around Nov, and postal_code features indicate that data might be coming from US.   
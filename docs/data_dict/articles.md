Total number of items 105,542

| name                         | type   | description                                                                                              |
|------------------------------|--------|----------------------------------------------------------------------------------------------------------|
| article_id                   | int    | unique item id (uniq product id x color)                                                                 | 
| product_code                 | int    | product id, non unique (47224 values)                                                                    |
| prod_name                    | string | product name, non unique (45875 values) some more frequent, prob diff colours                            |
| product_type_no              | int    | product type category (132 values) clothing part indicator (int codes for trousers, dress, sweater)      |
| product_type_name            | string | product type category (131 values) clothing part indicator name (trousers, dress, sweater)               |
| product_group_name           | string | prod group (19 values) more general prod category (upper body, lower body, accessories, underwear)       |
| graphical_appearance_no      | int    | id of graphical category (30 values)                                                                     |
| graphical_appearance_name    | string | appereance type category (solid, pring, denim) (30 values)                                               |
| colour_group_code            | int    | colour_group_code (50 values)                                                                            |
| colour_group_name            | string | colour_group_code (50 values) color names (black, grey, blue)                                            |
| perceived_colour_value_id    | int    | perceived_colour_value_id (8 values)                                                                     |
| perceived_colour_value_name  | string | perceived colour name (8 values)                                                                         |
| perceived_colour_master_name | string | perceived_colour_master_name (20 values) more general color name                                         |
| department_no | int    | department_id (299 values) dep id                                                                        |
| department_name | string | department_name (250 values) more detailed product category with Kids, Boy, Girl, Man, Women distinction |
| index_code| int    | index code                                                                                               | 
| index_name | string | even more general category (10 values) Ladies, Men, Children, Baby, Sport                                |
| index_group_name | string | even more more general category 5 values (Sport, Lady, Men, Baby, Divided)                               |
| section_name | string | quite detailed split of items (56 values)  collection name?                                              |
| garment_group_name | string | detailed prod category                                                                                   |
| detail_desc,object | string | prod description (43404 values) half missing                                                             |

Useful features for exploration (full history, 1 transaction, 3mo)
- index_group_name / index_name
- perceived_colour_master_name
- product_group_name (how many items of each category customers buy over 1mo/3mo/1y/full)
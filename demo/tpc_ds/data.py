from trilogy_public_models import get_executor
import pandas as pd

class DataProvider:
    def __init__(self, model:str):
        self.executor = get_executor(model)

    def fetch(self, query:str, columns:list[str]):
        return pd.DataFrame.from_records(self.executor.execute_query(query).fetchall(), columns=columns )

if __name__ == '__main__':
    x = DataProvider('duckdb.tpc_ds')

    df = x.fetch('''
select 
    web_sales.date.date,
    max(web_sales.external_sales_price) as web_sales.external_sales_price_max,
;
''', columns=['state_val', 'store_sales.store.state'])
    df['state_val'] = (df['state_val']).apply(pd.to_datetime, errors='coerce')
    print(df.head(10))
    print(df.dtypes)

#     df = x.fetch('''select 
#                  store_sales.store.state,
# sum(store_sales.sales_price) as state_val,

# order by
# store_sales.store.state asc
# ;''', columns=['state_val', 'store_sales.store.state'])
    
#     print(df.head(10))
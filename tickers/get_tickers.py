import pandas as pd
import os
import datetime as dt 
import yfinance as yf
import requests as req
from io import StringIO
from curl_cffi import requests
from utils.shared_lock import FILE_LOCK

base_dir = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(base_dir, "ticker_list.txt")
time_path = os.path.join(base_dir, "timestamp.txt")
batch_size = 10 

def fetch_nasdaq_tickers():
    url_nasdaq = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    url_others = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
    try:
        response1 = req.get(url_nasdaq, timeout=10)
        response2 = req.get(url_others, timeout=10)

        response1.raise_for_status() 
        response2.raise_for_status() 
        #raises exception if response unsuccessful

        lines1 = response1.text.strip().splitlines() 
        lines2 = response2.text.strip().splitlines() 
        #lines is a list of each row, also don't really need .strip()
        
        creation_time1 = lines1[-1][20:33]
        creation_time2 = lines2[-1][20:33]
        #last row for url_nasdaq is in form "File Creation Time: MMDDYYYYHH:MM|||||||", so 20:33 extracts the time

        date_format = "%m%d%Y%H:%M"
        date1 = dt.datetime.strptime(creation_time1, date_format)
        date2 = dt.datetime.strptime(creation_time2, date_format)

        data1 = "\n".join(lines1[:-1])
        data2 = "\n".join(lines2[:-1])

        df1 = pd.read_csv(StringIO(data1), sep="|")
        df2 = pd.read_csv(StringIO(data2), sep="|")
        #data in format "Symbol|Security Name|...|Test Issue|...|..."

        df1 = df1[(df1['Test Issue'] == 'N') & (df1.iloc[:, 0].notna())]
        df2 = df2[(df2['Test Issue'] == 'N') & (df2.iloc[:, 0].notna())]
        #'Test Issue' == 'Y' indicates a test ticker, so only == 'N'. 'Test Issue' col in different order for different files

        ticker_to_name_dict1 = dict(zip(df1.iloc[:, 0], df1.iloc[:, 1]))
        ticker_to_name_dict2 = dict(zip(df2.iloc[:, 0], df2.iloc[:, 1]))
        ticker_to_name_dict = ticker_to_name_dict1 | ticker_to_name_dict2
        #print(ticker_to_name_dict)

        all_tickers = df1["Symbol"].tolist() + df2["ACT Symbol"].tolist()
        #print(all_tickers)
        
        session = requests.Session(impersonate="chrome")
        #impersonate browswer incase we come into any issues regarding this

        unique_tickers = []

        #Test yf.download function 
        #yf_data = yf.download(['SRTY', 'SRV', 'SRVR', 'SRXH', 'SSB', 'SSD', 'SSFI', 'SSG', 'SSL', 'SSO', 'SSPX', 'SSPY', 'SST', 'SSTK', 'SSUS', 'SSXU', 'SSY', 'ST', 'STAG', 'STAX', 'STBF', 'STC', 'STCE', 'STE', 'STEL', 'STEM', 'STEW', 'STG', 'STHH', 'STIP', 'STK', 'STLA', 'STM', 'STN', 'STNG', 'STOT', 'STOX', 'STPZ', 'STR', 'STRV', 'STRW', 'STT', 'STT$G', 'STVN', 'STWD', 'STXD', 'STXE', 'STXG', 'STXI', 'STXK', 'STXM', 'STXS', 'STXT', 'STXV', 'STZ', 'SU', 'SUB', 'SUI', 'SUN', 'SUPL', 'SUPV', 'SURE', 'SURI', 'SUSA', 'SUZ', 'SVAL', 'SVIX', 'SVM', 'SVOL', 'SVT', 'SVV', 'SVXY', 'SW', 'SWAN', 'SWK', 'SWX', 'SWZ', 'SXC', 'SXI', 'SXQG', 'SXT', 'SYF', 'SYF$A', 'SYF$B', 'SYFI', 'SYK', 'SYLD', 'SYNB', 'SYNX', 'SYY', 'SZK', 'SZNE', 'T', 'T$A', 'T$C', 'TAC', 'TACK', 'TAFI', 'TAFL', 'TAFM', 'TAGG', 'TAGS', 'TAIL', 'TAK', 'TAL', 'TALO', 'TAN', 'TAP', 'TAP.A', 'TAPR', 'TAXF', 'TAXM', 'TAXX', 'TBB', 'TBBB', 'TBF', 'TBFC', 'TBFG', 'TBG', 'TBI', 'TBJL', 'TBLL', 'TBLU', 'TBN', 'TBT', 'TBUX', 'TBX', 'TCAF', 'TCAL', 'TCHP', 'TCI', 'TCPB', 'TD', 'TDC', 'TDEC', 'TDF', 'TDG', 'TDOC', 'TDS', 'TDS$U', 'TDS$V', 'TDTF', 'TDTT', 'TDV', 'TDVG', 'TDVI', 'TDW', 'TDY', 'TE', 'TE.W', 'TEAF', 'TEC', 'TECB', 'TECK', 'TECL'], period="5d", group_by="ticker", auto_adjust=True, threads=False)
        for i in range(0, len(all_tickers), batch_size):
            #splicing is safe for out of bounds, so don't need to worry about specifics of the last splice when not perfectly divisble by batch_size
            yf_data = yf.download(all_tickers[i:i+batch_size], period="5d", group_by="ticker", auto_adjust=True, threads=False, session=session)
            #yf_data is a MultiIndex with columns in the format of [(SYMBOL, VAR1), (SYMBOL, VAR2) ...]
            #the symbol name itself is the level at index 0, which we get with get_level_values(0), and we use.unique since each symbol has multiple columns for each corresponding VAR
            #if we fail to find a symbol, then we still have the columns, but the columns value will be NaN
            #...notna() turns all NaN to false, first .any() = True if any are true, first call is for all rows in each col, and second call is for all cols
            #all() would be a better metric, but issues arise with period=1d, when two dates might be available, but one is empty due to time constraint
            unique_tickers.extend(ticker for ticker in yf_data.columns.get_level_values(0).unique() if yf_data[ticker].notna().any().any())
            time.sleep(1)
            #10 tickers every 1-2 seconds prevents us from hitting rate limit

        unique_tickers = sorted(unique_tickers)

        #print(unique_tickers)

        with FILE_LOCK:
            with open(output_file_path, "w") as f:
                for ticker in unique_tickers:
                    f.write(f"{ticker} - {ticker_to_name_dict[ticker]}\n")

        with open(time_path, "w") as f:
            f.write(str(date2) if date2 > date1 else str(date1))

        return True

    except Exception as e:
        print(f"Failed to fetch NASDAQ tickers: {e}")
        return False
import numpy as np
import pandas as pd 
import re
import matplotlib.pyplot as plt
from unidecode import unidecode
from sklearn.linear_model import LinearRegression
import folium
import branca
from statsmodels.tsa.statespace.sarimax import SARIMAX
Datas=pd.read_csv("historical_air_quality_2021_en.csv",parse_dates=["Data Time S"])
print (Datas.shape)

# lam sach du lieu 
# 1.xoa cac dong và cot khong co gia tri va khong dung toi 
#1.1 xoa dong 2624-3097 va xoa cac ban sao
Datas = Datas.dropna(how='all')
Datas = Datas.drop_duplicates()
print (Datas.shape)

###1.2 xoa cot url 
Datas=Datas.drop(columns="Url")
###1.3 xoa cot Time Tz
Datas=Datas.drop(columns="Data Time Tz")

# Chuyển đổi cột thành chuỗi
Datas['Data Time S'] = Datas['Data Time S'].astype(str)
# Giả sử df['result'] là cột bạn muốn xóa tất cả các chữ cái
Datas['Data Time S'] = Datas['Data Time S'].apply(lambda x: re.sub('[a-zA-Z]', ' ', x))
# Chuyển đổi cột thành chuỗi và thay thế tất cả các dấu '+'
Datas['Data Time S'] = Datas['Data Time S'].astype(str).apply(lambda x: re.sub('\\+', ' ', x))

Datas['Data Time S'] = Datas['Data Time S'].str.split(' ', n=2, expand=True)[0] + ' ' + Datas['Data Time S'].str.split(' ', n=2, expand=True)[1]
# Chuyển đổi cột thành datetime
Datas['Data Time S'] = pd.to_datetime(Datas['Data Time S'])

#chuyen doi tat ca ten tram thanh khong dau
Datas['Station name'] = Datas['Station name'].astype(str).apply(lambda x: unidecode(x))
# print(Datas)

def extract_city(station_name):
    if 'Hanoi' in station_name or 'Ha Noi' in station_name :
        return 'Hanoi'
    elif 'HCMC' in station_name or 'Ho Chi Minh City' in station_name:
        return 'HCMC'
    elif 'Da Nang' in station_name:
        return 'Da Nang'
    elif 'Nha Trang' in station_name:
        return 'Khanh Hoa'
    elif 'Thua Thien Hue' in station_name:
        return 'Thua Thien Hue'
    elif 'Quang Ninh' in station_name or 'Ha Long' in station_name:
        return 'Quang Ninh'
    elif 'Bac Ninh' in station_name:
        return 'Bac Ninh'
    elif 'Gia Lai' in station_name:
        return 'Gia Lai'
    elif 'Lao Cai' in station_name:
        return 'Lao Cai'
    elif 'Cao Bang' in station_name:
        return 'Cao Bang'
    else:
        return None 
# Tạo cột mới "City" 
Datas['City'] = Datas['Station name'].apply(extract_city)

def extract_range(station_name):
    if 'Hanoi' in station_name or 'Ha Noi' in station_name or 'Quang Ninh' in station_name or 'Ha Long' in station_name or 'Lao Cai' in station_name or 'Cao Bang' in station_name or 'Bac Ninh' in station_name:
        return 'Bac'
    elif 'HCMC' in station_name or 'Ho Chi Minh City' in station_name or 'Gia Lai' in station_name:
        return 'Nam'
    elif 'Thua Thien Hue' in station_name or 'Nha Trang' in station_name or 'Da Nang' in station_name:
        return 'Trung'
    else:
        return None 
# Tạo cột mới "City" 
Datas['Region'] = Datas['Station name'].apply(extract_range)
print(Datas)

#sap xep lai theo thanh pho -ma tram - vi tri - ten tram để dễ dàng quan sát dữ liệu và sửa dối theo từng vùng từng thành phố  
Datas = Datas.sort_values(['City','Station ID', 'Location', 'Station name'])


columns_of_interest = ['CO', 'Dew', 'Humidity', 'NO2', 'O3', 'Pressure', 'PM10', 'PM2.5', 'SO2', 'Temperature','Wind']

for column in columns_of_interest:
    Datas[column] = np.where((Datas[column]=='-') | (Datas[column]=='No data') | (Datas[column]=='0') | (Datas[column]=='#NAME?'), None, Datas[column])


for column in columns_of_interest:
    Datas[column]=pd.to_numeric(Datas[column], errors='coerce')

correlation_matrix = Datas[columns_of_interest].corr()
print(correlation_matrix)

print(Datas.isnull().sum())


##############################################################################################################################

def fill_missing_values(Datas, cities, columns_of_interest):
    for city in cities:
        for column in columns_of_interest:
            mean_value = Datas[Datas['City'] == city][column].mean()
            if np.isnan(mean_value):  # Nếu thành phố không có giá trị trung bình cho cột
                # Tìm các thành phố khác cùng miền
                Region_of_city = Datas[Datas['City'] == city]['Region'].iloc[0]
                cities_same_range = Datas[Datas['Region'] == Region_of_city]['City'].unique()
                
                # Tính giá trị trung bình của cột từ các thành phố khác cùng miền
                mean_value = Datas[(Datas['City'].isin(cities_same_range)) & (Datas['City'] != city)][column].mean()
                
                # Nếu không có giá trị trung bình từ các thành phố khác cùng miền, sử dụng giá trị trung bình của tỉnh khác
                if np.isnan(mean_value):
                    mean_value = Datas[Datas['City'] != city][column].mean()       
            Datas.loc[(Datas['City'] == city) & (Datas[column].isnull()), column] = mean_value
    return Datas


correlation_matrix = Datas[columns_of_interest].corr()
print(correlation_matrix)

print(Datas['City'].unique())
cities = ['Bac Ninh', 'Cao Bang', 'Da Nang', 'Gia Lai', 'HCMC', 'Hanoi', 'Khanh Hoa', 'Lao Cai', 'Quang Ninh', 'Thua Thien Hue']
missing_co_rows = Datas[(Datas['CO'].isnull()) & (Datas['City'].isin(cities))]
Datas = fill_missing_values(Datas, cities, columns_of_interest)

print(Datas.head(10))


print(Datas.isnull().sum())

print(Datas.describe())

Datas['AQI index'] = np.where((Datas['AQI index']=='-'), np.nan, Datas['AQI index'])
# Tách dữ liệu thành hai phần: một phần có giá trị AQI và một phần không
df_with_aqi = Datas.dropna(subset=['AQI index'])
df_without_aqi = Datas[Datas['AQI index'].isna()]

# Tách dữ liệu có giá trị AQI thành tập huấn luyện và tập kiểm tra
X_train = df_with_aqi[['O3', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2']]
y_train = df_with_aqi['AQI index']

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán giá trị AQI cho dữ liệu không có giá trị AQI
X_pred = df_without_aqi[['O3', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2']]
Datas.loc[Datas['AQI index'].isna(), 'AQI index'] = model.predict(X_pred)


print(Datas['Dominent pollutant'].unique())
Datas['Dominent pollutant'] = Datas.apply(lambda row: 'pm10' if row['PM10'] > row['PM2.5'] else 'pm25', axis=1)

Datas['AQI index'] = pd.to_numeric(Datas['AQI index'], errors='coerce')
Datas['Alert level'] = Datas.apply(lambda row: '1' if row['AQI index'] < 50
                                    else ('2' if row['AQI index'] < 100 
                                    else ('3' if row['AQI index'] < 150 
                                    else ('4' if row['AQI index'] < 200 
                                    else ('5' if row['AQI index'] < 300 
                                    else '6')))), axis=1)

Datas['Status'] = Datas.apply(lambda row: 'Good' if row['AQI index'] < 50
                                    else ('Moderate' if row['AQI index'] < 100 
                                    else ('Unhealthy for sensitive groups' if row['AQI index'] < 150 
                                    else ('Unhealthy' if row['AQI index'] < 200 
                                    else ('Very hazardous to health' if row['AQI index'] < 300 
                                    else 'Dangerous')))), axis=1)
print(Datas['Dominent pollutant'][10:20])


#########################################################################################################################

#Phân tích Xu hướng:
Datas = Datas.sort_values(['Data Time S']) # vẽ biểu đồ theo thời gian 
Datas.rename(columns={'AQI index': 'AQI'}, inplace=True)

def plot_pollutants_by_Month(column, title, xlabel, ylabel):
    # Tạo một DataFrame tạm thời để không thay đổi DataFrame gốc
    temp = Datas.copy()
    temp.set_index('Data Time S', inplace=True)
    temp.index = pd.to_datetime(temp.index)
    # Tạo một cột mới 'Month' từ chỉ mục 'Data Time S'
    temp['Month'] = temp.index.month
    # Tính giá trị trung bình của chất cho mỗi tháng
    monthly_aqi = temp.groupby('Month')[column].mean()

    # Vẽ biểu đồ cột
    plt.figure(figsize=(10,6))
    monthly_aqi.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(f"{ylabel} (μg / m³)")
    plt.show()
print(Datas.columns)

print(Datas.describe())
columns1= ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2','AQI']
for col in columns1 :
    plot_pollutants_by_Month(col ,f"Chỉ số {col} trung bình theo từng tháng",'Month',col)

#Phân tích So sánh
# Tính giá trị trung bình của mỗi chất ô nhiễm cho mỗi thành phố
mean_values = Datas.groupby('City')[columns1].mean()

# Chuyển đổi DataFrame để mỗi thành phố là một cột và mỗi chất ô nhiễm là một hàng
unstacked_data = mean_values.unstack().reset_index()
unstacked_data.columns = ['Pollutant', 'City', 'Mean Value']

# Tạo bảng chéo giữa các thành phố và các chất ô nhiễm
pivot_table = unstacked_data.pivot(index='Pollutant', columns='City', values='Mean Value')

# Vẽ biểu đồ
pivot_table.plot(kind="bar", figsize=(10,10))
plt.ylabel('Mean Value')
plt.title('Average pollutant values by city')
plt.show()

# Tính giá trị trung bình của mỗi chất ô nhiễm cho mỗi khu vực
mean_values = Datas.groupby('Region')[columns1].mean()

# Vẽ biểu đồ
mean_values.plot(kind="bar", figsize=(10,10))
plt.ylabel('Mean Value')
plt.title('Average pollutant values by region')
plt.show()

# Chọn các cột tương ứng với các chất ô nhiễm
pollutants = Datas[['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'AQI']]

# Tính ma trận tương quan
correlation_matrix = pollutants.corr()

# In ma trận tương quan
print(correlation_matrix)

# Vẽ biểu đồ heatmap của ma trận tương quan
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()

# phân tích theo bản đồ địa lý 
# Tính AQI trung bình cho mỗi vị trí
avg_aqi = Datas.groupby('Location')['AQI'].mean().reset_index()

# Tạo bản đồ với tọa độ trung tâm là Việt Nam
m = folium.Map(location=[14.0583, 108.2772], zoom_start=6)

# Tạo một bảng màu
colorscale = branca.colormap.StepColormap(
    colors=['green', 'yellow', 'orange', 'red', 'purple', 'maroon'],
    index=[0, 51, 101, 151, 201, 301],
    vmin=0,
    vmax=500,
    caption='Air Quality Index (AQI)',
)
m.add_child(colorscale)


# Hàm để chuyển đổi AQI thành màu sắc tương ứng
def aqi_to_color(aqi):
    if aqi <= 50:
        return 'green'
    elif aqi <= 100:
        return 'yellow'
    elif aqi <= 150:
        return 'orange'
    else:
        return 'red'

# Thêm các điểm dữ liệu vào bản đồ
for index, row in avg_aqi.iterrows():
    # Tách cột 'location' thành vĩ độ và kinh độ
    lat, lon = map(float, row['Location'].split(','))
    aqi = row['AQI']
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        color=aqi_to_color(aqi),
        fill=True,
        fill_color=aqi_to_color(aqi),
        fill_opacity=0.6,
        popup=f'AQI ={aqi}',
    ).add_to(m)

# Hiển thị bản đồ
m.save('map.html')

df = pd.read_csv("aqi_airqualitydata_2020_en.csv", parse_dates=['Date'])

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")

# Set 'Date' as the index of the DataFrame
df.set_index('Date', inplace=True)

# List of substances
substances = ['co', 'pm10', 'no2', 'pm25', 'o3','so2', 'aqi']

# Calculate monthly medians for each substance
average_values = pd.DataFrame()
for substance in substances:
    substance_data = df[df['Specie'] == substance]
    average_values[substance] = substance_data.resample('ME')['median'].median()

# Filter data
average_values = average_values['2019-12':'2020-12']

# Interpolate missing values
average_values['aqi'] = average_values['aqi'].interpolate(method='linear')

# Rename columns
average_values.columns = average_values.columns.str.upper()
average_values.rename(columns={'PM25': 'PM2.5'}, inplace=True)

# Print results
print(average_values)

# Chuẩn bị dữ liệu bổ sung
columns1 = ['CO', 'PM10', 'NO2', 'PM2.5', 'O3','SO2', 'AQI']
Datas.rename(columns={'AQI index': 'AQI'}, inplace=True)
Datas.rename(columns={'Data Time S': 'Date'}, inplace=True)
Datas[columns1] = Datas[columns1].apply(pd.to_numeric, errors='coerce')

# Tạo một DataFrame tạm thời để không thay đổi DataFrame gốc
temp = Datas.copy()

# Chuyển đổi cột 'Data Time S' thành định dạng datetime và đặt nó làm chỉ mục
temp['Date'] = pd.to_datetime(temp['Date'])
temp.set_index('Date', inplace=True)

# Tính giá trị trung bình theo từng tháng của từng năm
monthly_aqi = temp.resample('ME')[columns1]

# Chuyển monthly_aqi thành DataFrame bằng cách gọi phương thức mean (hoặc một phương thức khác tùy thuộc vào nhu cầu của bạn)
monthly_aqi_df = monthly_aqi.mean()

# Bây giờ bạn có thể nối average_values và monthly_aqi_df
df = pd.concat([average_values, monthly_aqi_df])

# In kết quả
print(df)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train = df[:-1]
test = df[-1:]

# Danh sách các chất ô nhiễm
pollutants = ['CO', 'PM10', 'NO2', 'PM2.5', 'O3', 'SO2', 'AQI']
# Tạo một dictionary để lưu trữ dự đoán
predictions_dict = {}
# Xây dựng một mô hình SARIMAX cho mỗi chất ô nhiễm
for pollutant in pollutants:
    # Biến đổi dữ liệu bằng hàm log
    train_log = np.log1p(train[pollutant])

    # Xây dựng mô hình SARIMAX với dữ liệu đã biến đổi
    model = SARIMAX(train_log, order=(2, 1, 1), seasonal_order=(1, 2, 2, 12))
    model_fit = model.fit(disp=False)

    # Dự đoán giá trị cho tháng 12-2021
    predictions_log = model_fit.predict(len(train), len(df)-1)
    # Biến đổi ngược lại dự đoán bằng hàm exponen
    predictions = np.expm1(predictions_log)
    # In dự đoán
    predictions_dict[pollutant] = predictions[0]

# Vẽ biểu đồ từ dữ liệu dự đoán
plt.figure(figsize=(9,9))
plt.bar(predictions_dict.keys(), predictions_dict.values())
plt.xlabel('Pollutant')
plt.ylabel('Predicted Value for Dec 2021')
plt.title('Predicted Air Quality for Dec 2021')

plt.show()

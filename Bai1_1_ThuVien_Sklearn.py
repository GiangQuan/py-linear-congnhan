import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu
df = pd.read_csv('Tham Khao\Bai1_1\LinearRegression_CongNghiep.csv')

# Lấy dữ liệu đầu vào và đầu ra
X = df[['Số công nhân']]
y = df['Sản lượng/ngày (sản phẩm)']

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X, y)

# In hệ số góc và hệ số chặn
print("Hệ số góc (slope):", model.coef_[0])
print("Hệ số chặn (intercept):", model.intercept_)

# Vẽ biểu đồ hồi quy
plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, model.predict(X), color='red', label='Đường hồi quy')
plt.xlabel('Số công nhân')
plt.ylabel('Sản lượng/ngày (sản phẩm)')
plt.title('Hồi quy tuyến tính')
plt.legend()
plt.grid(True)
plt.show()

# Dự đoán với số công nhân = 35
du_doan_35 = model.predict([[35]])
print(f'Dự đoán sản lượng/ngày khi có 35 công nhân: {du_doan_35[0]:.2f}')

# Nhập từ bàn phím
so_cong_nhan = float(input("Nhập số công nhân: "))
du_doan = model.predict([[so_cong_nhan]])
print(f'Dự đoán sản lượng/ngày với {so_cong_nhan} công nhân là: {du_doan[0]:.2f}')

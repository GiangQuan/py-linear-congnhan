import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("Tham Khao\Bai1_1\LinearRegression_CongNghiep.csv")
print(df.head())

X = df[['Số công nhân']]
y = df['Sản lượng/ngày (sản phẩm)']

model = LinearRegression()
model.fit(X,y)

print ('Hệ số góc(slope): ', model.coef_[0])
print ('Hệ số chặn(intercept): ', model.intercept_)

plt.scatter(X,y,color ='blue',label='dữ liệu thực tế')
plt.plot(X,model.predict(X),color='red',label='Đường hồi quy')
plt.xlabel('Số công nhân')
plt.ylabel('Sản lượng/ngày (Sản phẩm)')
plt.title('Hồi quy tuyến tính')
plt.legend()
plt.grid(True)
plt.show()

du_doan_35 = model.predict([[35]])
print('Dự đoán sản lượng với số công nhân là 35: ', du_doan_35)

so_cong_nhan = float(input('Nhập số công nhân: '))
du_doan = model.predict([[so_cong_nhan]])
print(f'Sản lương/ngày với {so_cong_nhan} công nhân: ', du_doan)



 
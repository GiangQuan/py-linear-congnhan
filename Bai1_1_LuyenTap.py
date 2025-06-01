import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Tham Khao\Bai1_1\LinearRegression_CongNghiep.csv')
print(df.head())

X= df[['Số công nhân']]
y= df['Sản lượng/ngày (sản phẩm)']

model = LinearRegression()
model.fit(X,y)

print('He so goc (slope): ',model.coef_[0])
print('He so chan (intercept): ', model.intercept_)

plt.scatter(X,y,color='blue', label='Du lieu thuc te')
plt.xlabel('So cong nhan')
plt.ylabel('San luong/ngay')
plt.plot(X,model.predict(X),color='red',label='Duong hoi quy')
plt.title('Hoi quy tuyen tinh')
plt.legend()
plt.show()

du_doan_35 = model.predict([[35]])
print(f'Du doan san luong/ngay voi 35 cong nhan:  {du_doan_35[0]:.2f}')

so_cong_nhan = float(input('Nhap so cong nhan: '))
du_doan = model.predict([[so_cong_nhan]])
print(f'Du doan san luong/ngay voi {so_cong_nhan} cong nhan: {du_doan[0]:.2f}')
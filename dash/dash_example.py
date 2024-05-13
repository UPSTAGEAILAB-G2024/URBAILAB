# 필요한 라이브러리 임포트
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff




foundation_data = pd.read_csv('data/JBC_DATA/0507_EDA.csv')
houseprice_columns = ['매매지수', '전세지수', '월세지수', 'KB주택매매가격지수', 'KB주택전세가격지수',
       '아파트매매실거래가격지수', '아파트매매가격지수', '아파트전세가격지수', '아파트월세통합가격지수',]

productindex_columns = ['소비자물가지수(총지수)', '소비자물가지수(주택임차료)',
       '소비자물가지수(전세)', '소비자물가지수(월세)', '생산자물가지수', '수입물가지수', '수출물가지수']

interate_columns=['한국은행 기준금리',
       '정부대출금리', '국민주택채권1종(5년)', '기대인플레이션']

# 나눠서 그릴 공간 생성
fig_LT = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=("House prices","Commodity Index","Interest Rates"))

for column in houseprice_columns:
    if column == '아파트매매실거래가격지수':
        fig_LT.add_trace(go.Scatter(x=foundation_data['날짜'],y=foundation_data[column],
                              name=column,
                              mode='lines+markers'),
                 row=1, col=1)
    else:
        fig_LT.add_trace(go.Scatter(x=foundation_data['날짜'],y=foundation_data[column],
                              name=column),
                 row=1, col=1)

for column in productindex_columns:
    fig_LT.add_trace(go.Scatter(x=foundation_data['날짜'],y=foundation_data[column],
                              name=column),
                 row=2, col=1)

for column in interate_columns:
    fig_LT.add_trace(go.Scatter(x=foundation_data['날짜'],y=foundation_data[column],
                              name=column),
                 row=3, col=1)


one_df = foundation_data.drop(columns=['매매지수','전세지수','월세지수','KB주택매매가격지수',
                                       'KB주택전세가격지수','아파트매매가격지수','아파트전세가격지수',
                                       '아파트월세통합가격지수'])
one_df = one_df[['날짜', '아파트매매실거래가격지수','소비자물가지수(총지수)', '소비자물가지수(주택임차료)',
       '소비자물가지수(전세)', '소비자물가지수(월세)', '생산자물가지수', '수입물가지수', '수출물가지수']]


# '날짜' 열을 제외하고 상관관계 계산
correlation_matrix = one_df.drop(['날짜'], axis=1).corr()

# 상관관계 히트맵 생성
fig_hm = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns.tolist(),
    y=correlation_matrix.columns.tolist(),
    annotation_text=np.around(correlation_matrix.values, decimals=2),
    colorscale='Viridis',
    showscale=True
)
fig_hm.update_layout(title="변수 간 상관관계 히트맵", title_x=0.5, xaxis=dict(side='bottom'))


import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd

# 데이터 로드
Commodity_data = pd.read_csv('data/JBC_DATA/국제상품가격.csv')
Commodity_data['날짜'] = pd.to_datetime(Commodity_data['날짜'])
commodity_columns = Commodity_data.columns[1:]
melted_data = Commodity_data.melt(id_vars=['날짜'], value_vars=commodity_columns, var_name='원자재', value_name='가격')

NationalMoney_data = pd.read_csv('data/JBC_DATA/본원통화.csv')
NationalMoney_data['날짜'] = pd.to_datetime(NationalMoney_data['날짜'])

Short_IR_data = pd.read_csv('data/JBC_DATA/국제금리.csv')
Short_IR_data['날짜'] = pd.to_datetime(Short_IR_data['날짜'])

Market_IR_data = pd.read_csv('data/JBC_DATA/시장금리.csv')
Market_IR_data.dropna(inplace=True)
Market_IR_data['날짜'] = pd.to_datetime(Market_IR_data['날짜'])

Producer_PI_data = pd.read_csv('data/JBC_DATA/생산자물가지수.csv')
Producer_PI_data.dropna(inplace=True)
Producer_PI_data['날짜'] = pd.to_datetime(Producer_PI_data['날짜'])

Consumer_PI_data = pd.read_csv('data/JBC_DATA/소비자물가지수.csv')
Consumer_PI_data.dropna(inplace=True)
Consumer_PI_data['날짜'] = pd.to_datetime(Consumer_PI_data['날짜'])

Bank_LoanIR_New_data = pd.read_csv('data/JBC_DATA/예금은행대출금리(신규).csv')
Bank_LoanIR_New_data.dropna(inplace=True)
Bank_LoanIR_New_data['날짜'] = pd.to_datetime(Bank_LoanIR_New_data['날짜'])

Bank_LoanIR_Balance_data = pd.read_csv('data/JBC_DATA/예금은행대출금리(잔액).csv')
Bank_LoanIR_Balance_data.dropna(inplace=True)
Bank_LoanIR_Balance_data['날짜'] = pd.to_datetime(Bank_LoanIR_Balance_data['날짜'])


# Commodity_data에서 원자재 이름을 추출 (첫 번째 열인 '날짜'는 제외)
commodity_columns = Commodity_data.columns[1:]

# 데이터를 재구조화하여 '날짜', '원자재', '가격' 형식의 DataFrame 생성
melted_data = Commodity_data.melt(id_vars=['날짜'], value_vars=commodity_columns, var_name='원자재', value_name='가격')

# 애니메이션 막대 그래프 생성
fig_am = px.bar(melted_data, 
             x='원자재', 
             y='가격', 
             color='원자재', 
             animation_frame='날짜',
             range_y=[melted_data['가격'].min(), melted_data['가격'].max()],
             log_y=True,
             title='날짜별 원자재 가격 변동')




# 외부 스타일시트 추가 (Bootstrap)
external_stylesheets = ['https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
                        'https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap']# Noto Sans KR 폰트 추가
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# 레이아웃 설정
app.layout = html.Div([
    html.H1("부동산 가격 변수 EDA", className='text-center', style={'font-family': 'Noto Sans KR'}),
    dcc.Dropdown(
        id='graph-dropdown',
        options=[
            {'label': '원자재 그래프', 'value': 'commodity'},
            {'label': '통화량 그래프', 'value': 'money'},
            {'label': '국제금리 그래프', 'value': 'interest_rate'},
            {'label': '시장금리', 'value': 'market_rate'},
            {'label': '생산자물가', 'value': 'producer_price'},
            {'label': '소비자물가', 'value': 'consumer_price'},
            {'label': '신규대출금리', 'value': 'new_loan_rate'},
            {'label': '잔액대출금리', 'value': 'balance_loan_rate'}
        ],
        value='commodity'
    ),
    dcc.Graph(id='graph-output'),
    dcc.Graph(id='heatmap-output'),
    dcc.Graph(id='example-graph-4',figure=fig_am),
    html.Div([
        html.Div([
            dcc.Graph(
                id='example-graph-2',figure=fig_LT)],className='col-md-6'),
        html.Div([
            dcc.Graph(
                id='example-graph-3',figure=fig_hm)],className='col-md-6'),
    ], className='row'),
    
])



# 콜백 정의
@app.callback(
    [Output('graph-output', 'figure'),
     Output('heatmap-output', 'figure')],
    [Input('graph-dropdown', 'value')]
)

def update_graph(graph_type):
    # 기본 temp 데이터 프레임 설정 (예시로 아파트 매매 실거래가격지수 포함)
    temp = one_df[['날짜', '아파트매매실거래가격지수']]
    temp['날짜'] = pd.to_datetime(temp['날짜'])

    if graph_type == 'commodity':
        fig = px.line(Commodity_data, x='날짜', y=Commodity_data.columns[1:], title='원자재 가격 변동')
        merged_df = pd.merge(temp, Commodity_data, on='날짜', how='inner')
    elif graph_type == 'money':
        fig = px.line(NationalMoney_data, x='날짜', y=NationalMoney_data.columns[1:], title='통화량 변동')
        merged_df = pd.merge(temp, NationalMoney_data, on='날짜', how='inner')
    elif graph_type == 'interest_rate':
        fig = px.line(Short_IR_data, x='날짜', y=Short_IR_data.columns[1:], range_y=(-1,12.5), title='국제금리 변동')
        merged_df = pd.merge(temp, Short_IR_data, on='날짜', how='inner')
    elif graph_type == 'market_rate':
        fig = px.line(Market_IR_data, x='날짜', y=Market_IR_data.columns[1:], title='시장금리')
        merged_df = pd.merge(temp, Market_IR_data, on='날짜', how='inner')
    elif graph_type == 'producer_price':
        fig = px.line(Producer_PI_data, x='날짜', y=Producer_PI_data.columns[1:], title='생산자물가지수')
        merged_df = pd.merge(temp, Producer_PI_data, on='날짜', how='inner')
    elif graph_type == 'consumer_price':
        fig = px.line(Consumer_PI_data, x='날짜', y=Consumer_PI_data.columns[1:], title='소비자물가지수')
        merged_df = pd.merge(temp, Consumer_PI_data, on='날짜', how='inner')
    elif graph_type == 'new_loan_rate':
        fig = px.line(Bank_LoanIR_New_data, x='날짜', y=Bank_LoanIR_New_data.columns[1:], title='예금은행대출금리(신규)')
        merged_df = pd.merge(temp, Bank_LoanIR_New_data, on='날짜', how='inner')
    else:
        fig = px.line(Bank_LoanIR_Balance_data, x='날짜', y=Bank_LoanIR_Balance_data.columns[1:], title='예금은행대출금리(잔액)')
        merged_df = pd.merge(temp, Bank_LoanIR_Balance_data, on='날짜', how='inner')

    # 상관관계 히트맵 생성
    correlation_matrix = merged_df.drop(['날짜'], axis=1).corr()
    heatmap = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns.tolist(),
        y=correlation_matrix.columns.tolist(),
        annotation_text=np.around(correlation_matrix.values, decimals=2),
        colorscale='Viridis',
        showscale=True
    )
    heatmap.update_layout(title=f"{graph_type} 데이터와 아파트 매매 가격의 상관관계", title_x=0.5, xaxis=dict(side='bottom'))

    return fig, heatmap

# 서버 실행
if __name__ == '__main__':
    app.run_server(debug=True)



import dash
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd

# Dash 앱 초기화
app = dash.Dash(__name__)

data = pd.read_csv('data\JBC_DATA\sample.csv')
# 날짜 열을 datetime 타입으로 변환
data['날짜'] = pd.to_datetime(data['날짜'])

data_diff = data.diff().dropna() 
data_diff_2 = data_diff.diff().dropna()
# 모든 정상 시계열 데이터 선택 (음식 및 숙박 제외)
columns_to_use_all = [col for col in data_diff_2.columns if col != '음식 및 숙박']
data_model_all = data_diff_2[columns_to_use_all]

# VAR 모델 생성 및 적합
model_all = VAR(data_model_all.set_index('날짜'))
results_all = model_all.fit(maxlags=3, ic='aic')  # AIC 기준으로 최적의 lag 선택

# 플롯 생성 함수
def create_plots(col):
    coefficients = results_all.params[col].drop('const')
    p_values = results_all.pvalues[col].drop('const')
    labels = coefficients.index
    
    # 계수 플롯
    coef_fig = go.Figure()
    coef_fig.add_trace(go.Bar(y=labels, x=coefficients, orientation='h', marker=dict(color='skyblue')))
    coef_fig.update_layout(title_text=f'{col} - Regression Coefficients', xaxis_title='Coefficients')

    # p-value 플롯
    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    pval_fig = go.Figure()
    pval_fig.add_trace(go.Bar(y=labels, x=p_values, orientation='h', marker=dict(color=colors)))
    pval_fig.add_shape(type="line", x0=0.05, x1=0.05, y0=0, y1=len(p_values), line=dict(color="grey", width=2, dash="dot"))
    pval_fig.update_layout(title_text=f'{col} - p-values of Coefficients', xaxis_title='p-values')

    return coef_fig, pval_fig

# 앱 레이아웃 설정
app.layout = html.Div([
    html.H1("Regression Analysis"),
    dcc.Tabs([dcc.Tab(label=col, children=[
        dcc.Graph(figure=create_plots(col)[0]),
        dcc.Graph(figure=create_plots(col)[1])
    ]) for col in results_all.params.columns])
])

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)

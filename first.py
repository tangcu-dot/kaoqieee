# 第8章 企鹅分类项目
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')

# ========== 新增：模型训练和保存功能 ==========
def train_and_save_model():
    """
    训练随机森林模型并保存模型文件
    仅在模型文件不存在时运行
    """
    try:
        # 首先检查文件是否存在
        if not os.path.exists('penguins-chinese.csv'):
            st.error("❌ 找不到数据文件 'penguins-chinese.csv'")
            st.info("当前目录下的文件有：" + str(os.listdir('.')))
            return None, None
        
        st.info("正在读取数据文件...")
        
        # 尝试用不同编码读取文件（移除chardet依赖）
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'utf-8-sig', 'latin1', 'cp1252', 'ansi']
        
        df = None
        successful_encoding = None
        
        for enc in encodings_to_try:
            try:
                st.write(f"尝试用 {enc} 编码读取...")
                df = pd.read_csv('penguins-chinese.csv', encoding=enc)
                successful_encoding = enc
                st.write(f"✅ 使用 {enc} 编码读取成功!")
                break
            except UnicodeDecodeError as e:
                st.write(f"❌ {enc} 编码失败: Unicode解码错误")
                continue
            except Exception as e:
                st.write(f"❌ {enc} 编码读取错误: {e}")
                continue
        
        if df is None:
            st.error("无法用任何编码读取文件，请检查文件格式")
            st.write("尝试用二进制模式读取...")
            try:
                # 尝试不指定编码
                df = pd.read_csv('penguins-chinese.csv')
                successful_encoding = "自动检测"
                st.write(f"✅ 使用默认编码读取成功!")
            except Exception as e:
                st.error(f"完全无法读取文件: {e}")
                return None, None
        
        st.write(f"✅ 数据读取成功！编码: {successful_encoding}, 数据形状: {df.shape}")
        st.write("数据列名:", list(df.columns))
        
        # 处理缺失值
        st.info("处理缺失值...")
        original_rows = len(df)
        df = df.dropna()
        after_rows = len(df)
        st.write(f"删除缺失值: 从 {original_rows} 行到 {after_rows} 行")
        
        if len(df) == 0:
            st.error("数据为空，无法训练模型")
            return None, None
        
        # 显示数据前几行
        st.write("数据前5行:")
        st.dataframe(df.head())
        
        # 根据实际列名调整
        actual_columns = list(df.columns)
        
        # 显示所有列名帮助调试
        st.write("所有列名:", actual_columns)
        
        # 查找目标列（物种列）
        species_col = None
        possible_species_names = ['species', '物种', 'Species', 'label', 'target', 'class']
        
        for col in actual_columns:
            col_lower = col.lower()
            for possible in possible_species_names:
                if possible in col_lower:
                    species_col = col
                    break
            if species_col:
                break
        
        if not species_col:
            # 如果没有找到，尝试第一列可能是目标列
            st.warning("未找到明显的物种列，使用第一列作为目标列")
            species_col = actual_columns[0]
        
        st.write(f"目标列: {species_col}")
        
        # 查找特征列
        feature_cols = []
        numeric_cols = []
        categorical_cols = []
        
        for col in df.columns:
            if col != species_col:
                feature_cols.append(col)
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
        
        st.write(f"特征列: {feature_cols}")
        st.write(f"数值特征: {numeric_cols}")
        st.write(f"分类特征: {categorical_cols}")
        
        # 准备特征和目标变量
        X = df[feature_cols]
        y = df[species_col]
        
        # 对分类特征进行独热编码
        if categorical_cols:
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
        else:
            X_encoded = X
        
        st.write(f"编码后的特征形状: {X_encoded.shape}")
        
        # 对目标变量进行标签编码
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        st.write(f"目标变量类别: {list(le.classes_)}")
        st.write(f"类别编码: {dict(zip(le.transform(le.classes_), le.classes_))}")
        
        # 训练模型
        st.info("训练随机森林模型...")
        rfc_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc_model.fit(X_encoded, y_encoded)
        
        train_accuracy = rfc_model.score(X_encoded, y_encoded)
        st.write(f"✅ 模型训练完成！训练准确率: {train_accuracy:.2%}")
        
        # 保存模型
        with open('RFC_model.pkl', 'wb') as f:
            pickle.dump(rfc_model, f)
        
        # 创建并保存输出映射
        output_uniques_map = {i: species for i, species in enumerate(le.classes_)}
        with open('output_uniques.pkl', 'wb') as f:
            pickle.dump(output_uniques_map, f)
        
        st.success("✅ 模型训练完成并保存成功！")
        return rfc_model, output_uniques_map
        
    except Exception as e:
        st.error(f"❌ 模型训练失败: {str(e)}")
        import traceback
        st.write("详细错误信息:")
        st.code(traceback.format_exc())
        return None, None

# ========== 主要程序开始 ==========
# 设置页面的标题、图标和布局
st.set_page_config(
    page_title="企鹅分类器",  # 页面标题
    page_icon=":penguin:",  # 页面图标
    layout="wide",
)

# 使用侧边栏实现多页面显示效果
with st.sidebar:
    st.image('images/rigth_logo.png', width=100)
    st.title('请选择页面')
    page = st.selectbox("请选择页面", ["简介页面", "预测分类页面"], label_visibility='collapsed')

if page == "简介页面":
    st.title("企鹅分类器:penguin:")
    st.header('数据集介绍')
    st.markdown("""帕尔默群岛企鹅数据集是用于数据探索和数据可视化的一个出色的数据集，
也可以作为机器学习入门练习。
    该数据集是由 Gorman 等收集，并发布在一个名为 palmerpenguins 的 R 语言包，
以对南极企鹅种类进行分类和研究。
    该数据集记录了 344 行观测数据，包含 3 个不同物种的企鹅：阿德利企鹅、巴布亚企
鹅和帽带企鹅的各种信息。""")
    st.header('三种企鹅的卡通图像')
    st.image('images/penguins.png')

elif page == "预测分类页面":
    st.header("预测企鹅分类")
    st.markdown("这个 Web 应用是基于帕尔默群岛企鹅数据集构建的模型。只需输入 6 个信息，就可以预测企鹅的物种，使用下面的表单开始预测吧！")
    
    # 检查模型文件是否存在，如果不存在则训练模型
    if not os.path.exists('RFC_model.pkl') or not os.path.exists('output_uniques.pkl'):
        st.warning("⚠️ 未找到模型文件，正在训练新模型...")
        with st.spinner("正在训练模型，请稍候..."):
            rfc_model, output_uniques_map = train_and_save_model()
        if rfc_model is None:
            st.error("❌ 模型训练失败，请检查数据文件")
            st.stop()
    else:
        try:
            with open('RFC_model.pkl', 'rb') as f:
                rfc_model = pickle.load(f)
            with open('output_uniques.pkl', 'rb') as f:
                output_uniques_map = pickle.load(f)
            st.success("✅ 模型文件加载成功！")
        except Exception as e:
            st.error(f"❌ 模型加载失败: {str(e)}")
            st.stop()

    # 该页面是3:1:2的列布局
    col_form, col, col_logo = st.columns([3, 1, 2])
    
    with col_form:
        # 运用表单和表单提交按钮
        with st.form('user_inputs'):
            st.write("根据您的模型特征，请提供以下信息：")
            
            # 获取模型特征名
            feature_names = list(rfc_model.feature_names_in_)
            st.write(f"模型特征 ({len(feature_names)}个): {feature_names}")
            
            # 简化输入：只输入数值特征
            bill_length = st.number_input('喙长度（毫米）', min_value=30.0, max_value=70.0, value=40.0)
            bill_depth = st.number_input('喙深度（毫米）', min_value=10.0, max_value=25.0, value=18.0)
            flipper_length = st.number_input('翅膀长度（毫米）', min_value=170.0, max_value=240.0, value=200.0)
            body_mass = st.number_input('身体质量（克）', min_value=2000.0, max_value=7000.0, value=4000.0)
            
            # 添加分类特征输入
            island = st.selectbox('岛屿', options=['托尔德岛', '比斯科岛', '德里斯科岛'])
            sex = st.selectbox('性别', options=['雄性', '雌性', '未知'])
            
            submitted = st.form_submit_button('预测分类')

        if submitted:
            try:
                # 根据特征名称构建输入数据
                input_data = []
                
                # 为每个特征名构建对应的输入值
                for feature in feature_names:
                    feature_lower = feature.lower()
                    
                    # 匹配数值特征
                    if 'bill_length' in feature_lower or 'culmen_length' in feature_lower or '嘴峰' in feature:
                        input_data.append(bill_length)
                    elif 'bill_depth' in feature_lower or 'culmen_depth' in feature_lower or '嘴厚' in feature:
                        input_data.append(bill_depth)
                    elif 'flipper' in feature_lower or '鳍状肢' in feature or '翅膀' in feature:
                        input_data.append(flipper_length)
                    elif 'body_mass' in feature_lower or '体重' in feature or '质量' in feature:
                        input_data.append(body_mass)
                    # 匹配岛屿特征
                    elif 'torgersen' in feature_lower or '托尔' in feature:
                        input_data.append(1 if island == '托尔德岛' else 0)
                    elif 'biscoe' in feature_lower or '比斯科' in feature:
                        input_data.append(1 if island == '比斯科岛' else 0)
                    elif 'dream' in feature_lower or '德里斯科' in feature or '梦' in feature:
                        input_data.append(1 if island == '德里斯科岛' else 0)
                    # 匹配性别特征
                    elif 'male' in feature_lower or '雄性' in feature or '男' in feature:
                        input_data.append(1 if sex == '雄性' else 0)
                    elif 'female' in feature_lower or '雌性' in feature or '女' in feature:
                        input_data.append(1 if sex == '雌性' else 0)
                    else:
                        # 默认值
                        input_data.append(0)
                
                format_data_df = pd.DataFrame(data=[input_data], columns=feature_names)
                
                # 显示输入数据用于调试
                st.write("输入数据:")
                st.write(format_data_df)
                
                # 使用模型进行预测
                predict_result_code = rfc_model.predict(format_data_df)
                
                # 将类别代码映射到具体的类别名称
                if predict_result_code[0] in output_uniques_map:
                    predict_result_species = output_uniques_map[predict_result_code[0]]
                else:
                    predict_result_species = f"类别 {predict_result_code[0]}"
                
                st.success(f'根据您输入的数据，预测该企鹅的物种名称是：**{predict_result_species}**')
                
                with col_logo:
                    try:
                        st.image(f'images/{predict_result_species}.png', width=300, caption=predict_result_species)
                    except:
                        st.image('images/rigth_logo.png', width=300, caption="企鹅分类器")
                
            except Exception as e:
                st.error(f"预测出错: {str(e)}")
                import traceback
                st.write("详细错误:")
                st.code(traceback.format_exc())
                st.info("请确保所有输入值都有效，然后重试。")

    # 显示在表单外
    with col_logo:
        if not submitted:
            st.image('images/rigth_logo.png', width=300, caption="请输入参数进行预测")

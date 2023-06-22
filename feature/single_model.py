'''
  @ Date: 2023/2/9 20:22
  @ Author: Zhao YaChen
'''
import csv
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier, StackingCVClassifier
from numpy import interp
from sklearn import svm, model_selection
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from myargs import args
import radiomics.featureextractor as FEE
import matplotlib.font_manager as fm

MODEL_TYPE = "ph3"
SAVE_MODEL = False
SAVE_RESULT = False

default_prefix = 'D:/Desktop/BREAST/BREAST/'
radiomics_path = '../case1/breast_input_'+MODEL_TYPE+'.csv'
new_radiomics_path = '../case3/breast_input_ph0_ph3_t2.csv'
test_radiomics_path = '../case1/' + 'breast_input_test_'+MODEL_TYPE+'.csv'

#保存模型的地址
train_model_path = "../case1/model/train_model_"+MODEL_TYPE
std_path = "../case1/model/std_"+MODEL_TYPE+".m"
selector_path = "../case1/model/selector_"+MODEL_TYPE+".m"
selector_rf_path = "../case1/model/feature_"+MODEL_TYPE+".m"

root_path = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData'

"""
    case1-将单个模态csv文件进行特征降维
"""
# 特征降维并针对数据进行SVM分析
def dimensionality_reduction(way, csv_root, save_model=False, is_less_count_feature=False):
    if os.path.exists(csv_root):
        data = pd.read_csv(csv_root)
        Y = data['pCR_label']
        data = data.iloc[:, 24: -1]
        X = np.array(data[data.columns])
        print('X.shape', X.shape)
        # 对特征数据进行标准化处理
        sc = StandardScaler().fit(X)
        X_std = sc.transform(X)
        print('X_std.shape', X_std.shape)
        if save_model:
            joblib.dump(sc, std_path)
        if way == 'RF':
            # ================================================单变量特征选择（Univariate Feature Selection）卡方检验================================================
            if is_less_count_feature:
                selector = SelectKBest(f_classif, k=60)
                X_new = selector.fit_transform(X_std, Y)
                print('X_new.shape after f_classif', X_new.shape)
                # 根据pvalues_画图
                scores = -np.log10(selector.pvalues_)
                scores /= scores.max()
                X_indices = np.arange(X_std.shape[-1])
                plt.bar(X_indices - .45, scores, width=.2,
                        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
                        edgecolor='black')
                if save_model:
                    joblib.dump(selector, selector_path)
                    plt.savefig('../case1/pic/kf_pvalues_'+MODEL_TYPE+'.png')
                plt.show()
                # 求特征
                scores = selector.scores_
                feature_kf_count = X_new.shape[-1]
                indices = np.argsort(-scores)[:feature_kf_count]
                indices.sort()
                data_columns_after_f = [data.columns[i] for i in indices]
            else:
                X_new = X_std
                data_columns_after_f = data.columns
            # ================================================随机森林================================================
            model = RandomForestRegressor(random_state=1)
            model.fit(X_new, Y)
            selector_rf = SelectFromModel(model)
            X_fit = selector_rf.fit_transform(X_new, Y)
            print('X_fit.shape after RF', X_fit.shape)
            # 拟合模型后，根据特征的重要性绘制成图
            importances = model.feature_importances_
            feature_final_count = X_fit.shape[-1]
            indices = np.argsort(importances)[-feature_final_count:]  # top 10 features
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [data_columns_after_f[i] for i in indices])
            plt.xlabel('Relative Importance')
            if save_model:
                joblib.dump(selector_rf, selector_rf_path)
                plt.savefig('../case1/pic/feature_importances_' + MODEL_TYPE + '.png')
            plt.show()
            # 求最终特征
            indices.sort()
            final_feature = [data_columns_after_f[i] for i in indices]
            for id_, item in enumerate(final_feature):
                print('第', str(id_), '个被选中的特征为:', item)
            return X_fit, Y, final_feature
    else:
        print('请输入正确的csv文件地址')
        return
"""
    case1-SVM分类器
"""
def svm_thyroid(X, y, k, save_model=False, save_result=False):
    model = svm.SVC(probability=True)  # 创建SVM分类器
    k_cv(model, 'SVM', X, y, k, save_model, save_result)
"""
    case1-公用函数
    画ROC曲线图
"""
def k_cv(model, model_name, X, y, k, save_model=False, save_result=False):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    accs = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        # 带概率的如下，不带概率的函数是model.predict(X[test])
        probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        # 根据probas_得到y_pred
        y_pred = np.argmax(probas_, axis=1)
        if save_result:
            y[test].to_csv('../case1/result/y_test_' + str(i) + 'th_fold.csv')
            with open('../case1/result/result_' + model_name + '_' + str(i) + 'th_fold_' + MODEL_TYPE + '_yproba.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['label0_proba', 'label1_proba'])
                for y_item in probas_:
                    writer.writerow(y_item)
            # with open('../case1/result/result_' + model_name + '_' + str(i) + 'th_fold_' + MODEL_TYPE + '_y.csv', 'w') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(['pid', 'predict_pCR'])
            #     for item, y_item in zip(y[test].items(), y_pred):
            #         writer.writerow([item[0], y_item])
        # 函数1：根据label和概率值求fpr和tpr，即Receiver operating characteristic(ROC)曲线
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        # 插值后的tpr值
        tprs.append(interp(mean_fpr, fpr, tpr))
        # 原点坐标0，0
        tprs[-1][0] = 0.0
        # 函数2：计算ROC曲线下面积
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # 计算准确率accuracy
        acc = accuracy_score(y[test], y_pred)
        accs.append(acc)
        print('accuracy is %s' % acc)
        # 计算精度precision
        print('precision is %s' % precision_score(y[test], y_pred))
        # 计算召回率recall
        print('recall is %s' % recall_score(y[test], y_pred, average='macro'))
        # 计算f1-score F1-score
        print('F1-score is %s' % f1_score(y[test], y_pred, average='macro'))
        # 画一条线
        plt.plot(fpr, tpr, linewidth=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f ACC = %0.2f)' % (i, roc_auc, acc))
        i += 1
    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='r', label='Chance', alpha=.8)
    # 计算平均tpr
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # 计算平均AUC值
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    # 计算平均accuracy
    mean_accuracy = np.mean(accs)
    std_accuracy = np.std(accs)
    # 画平均的线
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC ('
                   r'AUC = %0.2f $\pm$ %0.2f '
                   r'ACC = %0.2f $\pm$ %0.2f)'
                   % (mean_auc, std_auc, mean_accuracy, std_accuracy),
             linewidth=2, alpha=.8)
    # 填充阴影部分
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    # 坐标轴
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    # 图例
    plt.legend(loc="lower right")
    if save_model:
        joblib.dump(model, train_model_path + "_" + model_name + '.m')
        plt.savefig('../case1/pic/roc_' + model_name + '_' + str(k) + 'folds_' + MODEL_TYPE + '.png')
    plt.show()

"""
    case2-画ROC曲线图，输入可以是csv数据
"""
def k_cv_from_file(y_test_paths, probas_paths, labels):
    accs = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    y_test_list = []
    probas_list = []
    for y_test_path, probas_path in zip(y_test_paths, probas_paths):
        y_test_list.append(pd.read_csv(y_test_path)['pCR_label'])
        probas_list.append(pd.read_csv(probas_path).values)


    # https://zhuanlan.zhihu.com/p/403770925
    # 控制字体与字体大小
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    # plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.variant'] = 'normal'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['font.stretch'] = 'normal'
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # plt.rcParams['font.sans-serif'] = [u'SimSun']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    font0 = {'fontfamily': 'serif',
             'fontname': 'Times New Roman',
             'fontstyle': 'normal',
             'fontvariant': 'normal',
             'fontweight': 'bold',
             'fontstretch': 'normal',
             'fontsize': 12}
    zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
    # 固定了图片大小
    fw = 13.5 / 2.54
    fh = 11.5 / 2.54
    fig, ax = plt.subplots(num=1, figsize=(fw, fh))

    # 最后这一行调整画框的位置，用来消除白边。
    # plt.subplots_adjust(right=0.99, left=0.125, bottom=0.14, top=0.975)
    plt.subplots_adjust(left=0.11, right=0.97, bottom=0.11, top=0.97)
    colors = ['#FF0000', '#00FF00', '#0000FF', 'darkorange']
    i = 0
    for y_test, probas_, label in zip(y_test_list, probas_list, labels):
        # 函数1：根据label和概率值求fpr和tpr，即Receiver operating characteristic(ROC)曲线
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        # 插值后的tpr值
        tprs.append(interp(mean_fpr, fpr, tpr))
        # 原点坐标0，0
        tprs[-1][0] = 0.0
        # 函数2：计算ROC曲线下面积
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # 画一条线
        if i == 3:
            ax.plot(fpr, tpr, linewidth=1.5, color=colors[i], alpha=.9, label='本文方法$\mathrm{ (AUC = 0.89)}$')
            # ax.plot(fpr, tpr, linewidth=1.5, alpha=.9, label='本文方法$\mathrm{ (AUC = 0.89)}$')
        else:
            ax.plot(fpr, tpr, linewidth=1.5, color=colors[i], alpha=.9,  label='$\mathrm{%s (AUC = %0.2f)}$' % (label, roc_auc))
            # ax.plot(fpr, tpr, linewidth=1.5, alpha=.9, label='$\mathrm{%s (AUC = %0.2f)}$' % (label, roc_auc))
        # 根据probas_得到y_pred
        y_pred = np.argmax(probas_, axis=1)
        # 计算准确率accuracy
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        print('accuracy is %s' % acc)
        # 计算精度precision
        print('precision is %s' % precision_score(y_test, y_pred))
        # 计算召回率recall
        print('recall is %s' % recall_score(y_test, y_pred, average='macro'))
        # 计算f1-score F1-score
        print('F1-score is %s' % f1_score(y_test, y_pred, average='macro'))
        i += 1
    # 画对角线
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color='r', label='$\mathrm{Chance}$', alpha=.8)
    # 利用网格代替刻度
    ax.xaxis.grid(True, which='major', lw=0.5, linestyle='--', color='0.8', zorder=1)
    ax.yaxis.grid(True, which='major', lw=0.5, linestyle='--', color='0.8', zorder=1)
    # 坐标轴
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('False Positive Rate', fontdict=font0, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontdict=font0, fontweight='bold')
    # ax.set_title('Receiver operating characteristic example')
    # 图例
    legend = ax.legend(loc="lower right",
                       fancybox=False,
                       edgecolor='k',prop=zh_font)
    frame = legend.get_frame()
    frame.set_linewidth(0.57)
    # plt.savefig('../case2/pic/roc_single_sequence.png', dpi=300)
    plt.savefig('../case2/pic/roc_ensemble_stacking.png', dpi=300)
    plt.show()


"""
    case1-RF分类器
"""
def randomForest_thyroid(image,label, k, save_model=False, save_result=False):
    model = RandomForestClassifier(n_jobs=-1,max_features= 'auto' ,n_estimators=250, oob_score = True)
    k_cv(model, 'RF', image, label, k, save_model, save_result)

"""
    case1-预测
"""
def predict(ori_dir, model_name, save_csv_file=False, from_csv_file=False):
    if from_csv_file:
        data = pd.read_csv(test_radiomics_path)
    else:
        # 获取特征 使用配置文件初始化特征抽取器
        extractor = FEE.RadiomicsFeatureExtractor(args.params_file)
        if MODEL_TYPE == "t2":
            image_path = os.path.join(root_path, ori_dir, ori_dir + "_" + MODEL_TYPE + "_sitk.nii")
        else:
            image_path = os.path.join(root_path, ori_dir, ori_dir + "_" + MODEL_TYPE + ".nii")
        seg_path = os.path.join(root_path, ori_dir, ori_dir + "_seg.nii")
        result = extractor.execute(image_path, seg_path)
        if save_csv_file:
            save_df = pd.DataFrame([result])
            save_df.to_csv(test_radiomics_path, index=None, header=None)
        # 从字典创建DataFrame
        data = pd.DataFrame([result])
    data = data.iloc[:, 22:]
    X = np.array(data[data.columns], dtype=float)
    # 查看一下x的维度
    print('X.shape', X.shape)
    # 对特征数据进行标准化处理
    sc = joblib.load(std_path)
    X_std = sc.transform(X)
    print('X_std.shape', X_std.shape)
    print(X_std)
    # 单变量分析
    selector = joblib.load(selector_path)
    X_new = selector.transform(X_std)
    print('X_new.shape after f_classif', X_new.shape)
    # 随机森林多变量分析
    feature = joblib.load(selector_rf_path)
    X_fit = feature.transform(X_new)
    print('X_fit.shape after RF', X_fit.shape)
    # 开始预测 模型从本地调回
    svm = joblib.load(train_model_path + "_" + model_name + '.m')
    predict_result = svm.predict(X_fit)
    predict_result_proda = svm.predict_proba(X_fit)
    print('predict_result:', predict_result)
    print('predict_result_proda:', predict_result_proda)
    return X_fit, predict_result, predict_result_proda
"""
    case1-利用单个模态csv文件训练出的分类器
    手动进行硬投票
"""
def model_voting(model_name1, model_name2=None):
    ph0_acc = []
    ph3_acc = []
    t2_acc = []
    voting_acc = []
    for i in range(3):
        # read y_test
        y_test_csv = pd.read_csv("../case1/result/y_test_" + str(i) + "th_fold.csv")
        y_test = y_test_csv['pCR_label']
        # hard voting
        ph0_submission = pd.read_csv("../case1/result/result_" + model_name1 + "_" + str(i) + "th_fold_ph0_y.csv")
        ph3_submission = pd.read_csv("../case1/result/result_" + model_name1 + "_" + str(i) + "th_fold_ph3_y.csv")
        t2_submission = pd.read_csv("../case1/result/result_" + model_name1 + "_" + str(i) + "th_fold_t2_y.csv")
        ph0_result = ph0_submission['predict_pCR']
        ph3_result = ph3_submission['predict_pCR']
        t2_result = t2_submission['predict_pCR']
        all_data = [ph0_result, ph3_result, t2_result]
        if model_name2!= None:
            ph0_submission = pd.read_csv("../case1/result/result_" + model_name2 + "_" + str(i) + "th_fold_ph0_y.csv")
            ph3_submission = pd.read_csv("../case1/result/result_" + model_name2 + "_" + str(i) + "th_fold_ph3_y.csv")
            t2_submission = pd.read_csv("../case1/result/result_" + model_name2 + "_" + str(i) + "th_fold_t2_y.csv")
            ph0_result = ph0_submission['predict_pCR']
            ph3_result = ph3_submission['predict_pCR']
            t2_result = t2_submission['predict_pCR']
            all_data.append(ph0_result)
            all_data.append(ph3_result)
            all_data.append(t2_result)
            all_data.append(ph3_result)
        votes = pd.concat(all_data, axis='columns')
        predictions = votes.mode(axis='columns').to_numpy()
        voting_result = predictions.flatten()
        ph0_acc.append(accuracy_score(y_test, ph0_result))
        ph3_acc.append(accuracy_score(y_test, ph3_result))
        t2_acc.append(accuracy_score(y_test, t2_result))
        voting_acc.append(accuracy_score(y_test, voting_result))
        # 用于检验合并结果
        # all_data2 = [ph0_result, ph3_result, t2_result, y_test, pd.Series(predictions.flatten())]
        # votes2 = pd.concat(all_data2, axis='columns')
        # TODO: soft voting
    print('accuracy is %s %s %s %s' % (np.mean(ph0_acc), np.mean(ph3_acc), np.mean(t2_acc), np.mean(voting_acc)))

"""
    case1-输入是单个模态的csv文件，输出是准确率
"""
def case1_classification_acc(is_less_count_feature=False):
    data = pd.read_csv(radiomics_path)
    Y = data['pCR_label']
    data = data.iloc[:, 24: -1]
    X = np.array(data[data.columns], dtype=float)
    print('X.shape', X.shape)
    sc = StandardScaler().fit(X)
    X_std = sc.transform(X)
    print('X_std.shape', X_std.shape)
    if is_less_count_feature:
        selector = SelectKBest(f_classif, k=60)
        X_new = selector.fit_transform(X_std, Y)
        print('X_new.shape after f_classif', X_new.shape)
    else:
        X_new = X_std
    model = RandomForestRegressor(random_state=1)
    model.fit(X_new, Y)
    selector_rf = SelectFromModel(model)
    X_fit = selector_rf.fit_transform(X_new, Y)
    print('X_fit.shape after RF', X_fit.shape)
    clf1 = LogisticRegression(random_state=1)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=1)
    clf4 = RandomForestClassifier(random_state=1)
    clf5 = svm.SVC(random_state=1, probability=True)
    labels = ['Logistic Regression', 'KNN', 'Naive Bayes', 'Random Forest', 'SVM']

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5], labels):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        scores = model_selection.cross_val_score(clf, X_fit, Y,
                                                 cv=cv,
                                                 scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))

"""
    case3-输入是三个模态的竖版csv文件，输出是准确率
    ensemble文档:
    http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/#example-3-majority-voting-with-classifiers-trained-on-different-feature-subsets
"""
def case3_classification_acc():
    df = pd.read_csv(new_radiomics_path)
    Y = df['dceph0_pCR_label']
    except_idxs = df.columns.str.contains(r'diagnostics|pid|modal|pCR_label')
    df = df.iloc[:, ~except_idxs]
    X = np.array(df[df.columns], dtype=float)
    print('X.shape', X.shape)
    sc = StandardScaler().fit(X)
    X_std = sc.transform(X)
    print('X_std.shape', X_std.shape)
    # 卡方影响不大
    selector = SelectKBest(f_classif, k=60)
    X_new = selector.fit_transform(X_std, Y)
    print('X_new.shape after f_classif', X_new.shape)
    # 随机森林有提升
    model = RandomForestRegressor(random_state=1)
    model.fit(X_std, Y)
    selector_rf = SelectFromModel(model)
    X_fit = selector_rf.fit_transform(X_std, Y)
    print('X_fit.shape after RF', X_fit.shape)

    lr = LogisticRegression()
    clf1 = LogisticRegression(random_state=1)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=1)
    clf4 = RandomForestClassifier(random_state=1)
    clf5 = svm.SVC(random_state=1, probability=True)
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf4, clf5], weights=[1, 2, 1], voting='hard')
    sclf = StackingCVClassifier(classifiers=[clf1, clf4, clf5], meta_classifier=lr, random_state=1, use_probas=True)
    labels = ['Logistic Regression', 'KNN', 'Naive Bayes', 'Random Forest', 'SVM', 'Ensemble', 'StackingClassifier']

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf, sclf], labels):
        scores = model_selection.cross_val_score(clf, X_fit, Y,
                                                 cv=5,
                                                 scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))


if __name__ == '__main__':
    # ==========================expriment-case1:=================================================
    # ---------------------可以用这个估算下不同分类器准确率---------------------------------------------
    # case1_classification_acc(is_less_count_feature=True)
    # ---------------------首先运行’ph0‘'ph3''t2'各一次，注意SAVE等参数-------------------------------
    # X, Y, final_feature = dimensionality_reduction('RF', radiomics_path, save_model=SAVE_MODEL, is_less_count_feature=True)
    # svm_thyroid(X, Y, 3, save_model=SAVE_MODEL, save_result=SAVE_RESULT)
    # randomForest_thyroid(X, Y, 3, save_model=SAVE_MODEL, save_result=SAVE_RESULT)
    # ---------------------然后运行’SVM‘'RF'各一次-------------------------------------------------
    # model_voting('SVM', 'RF')
    # ---------------------最后进行prediction-case1-----------------------------------------------
    # ori_dir = "Breast_Training_005"
    # X_test, predict_result, predict_result_proda = predict(ori_dir, 'SVM', from_csv_file=False)
    # X_test, predict_result, predict_result_proda = predict(ori_dir, 'RF', from_csv_file=False)

    # ==========================expriment-case2:=================================================
    # # ---------------------用这个画不同方法的ROC图---------------------------------------------------
    # y_test_paths = [
    #     '../case1/result/y_test_2th_fold.csv',
    #     '../case1/result/y_test_1th_fold.csv',
    #     '../case1/result/y_test_1th_fold.csv',
    #     '../case1/result/y_test_0th_fold.csv',
    # ]
    # probas_paths = [
    #     '../case1/result/result_SVM_2th_fold_ph3_yproba.csv',
    #     '../case1/result/result_SVM_1th_fold_ph3_yproba.csv',
    #     '../case1/result/result_SVM_1th_fold_t2_yproba.csv',
    #     '../case1/result/result_RF_0th_fold_t2_yproba.csv',
    # ]
    # labels = ['S0', 'S3', 'T2', u'多序列']
    # k_cv_from_file(y_test_paths, probas_paths, labels)

    y_test_paths = [
        '../case1/result/y_test_2th_fold.csv',
        '../case1/result/y_test_0th_fold.csv',
        '../case1/result/y_test_2th_fold.csv',
        '../case1/result/y_test_0th_fold.csv',
    ]
    probas_paths = [
        '../case1/result/result_RF_2th_fold_ph3_yproba.csv',
        '../case1/result/result_SVM_0th_fold_t2_yproba.csv',
        '../case1/result/result_RF_2th_fold_t2_yproba.csv',
        '../case1/result/result_RF_0th_fold_t2_yproba.csv',
    ]
    labels = ['Hard  Voting', 'Soft  Voting', 'Stacking', '本文方法']
    k_cv_from_file(y_test_paths, probas_paths, labels)

    # ==========================expriment-case3:=================================================
    # case3_classification_acc()

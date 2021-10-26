import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


def logistic_regression():
    df = pd.read_csv('../Dataset/00 df.csv')
    train = df[df['flag'] == 'train']
    test = df[df['flag'] == 'test']


    cat_feats = ['age_bin', 'capital_gl_bin', 'education_bin', 'hours_per_week_bin', 'msr_bin', 'occupation_bin', 'race_sex_bin']

    y_train = train['y']
    x_train = train[['age_bin', 'capital_gl_bin', 'education_bin', 'hours_per_week_bin', 'msr_bin', 'occupation_bin', 'race_sex_bin']]
    x_train = pd.get_dummies(x_train, columns=cat_feats, drop_first=True)

    y_test = test['y']
    x_test = test[['age_bin', 'capital_gl_bin', 'education_bin', 'hours_per_week_bin', 'msr_bin', 'occupation_bin', 'race_sex_bin']]
    x_test = pd.get_dummies(x_test, columns=cat_feats, drop_first=True)


    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)


    test_calc = pd.concat([pd.DataFrame(y_test).reset_index(drop=True), pd.DataFrame(y_pred).reset_index(drop=True)], axis=1)
    test_calc.rename(columns={0: 'predicted'}, inplace=True)

    test_calc['predicted'] = test_calc['predicted'].apply(lambda x: 1 if x > 0.5 else 0)

    # ------------------------------------ Checking model performance using scikit -------------------------------------

    df_table = confusion_matrix(test_calc['y'], test_calc['predicted'])
    print("\n\n> Confusion Matrix Table: \n", df_table)

    pscore = precision_score(test_calc['y'], test_calc['predicted'], average='binary')
    print("\n\n> Precision Score ===>>>> ", pscore)

    rscore = recall_score(test_calc['y'], test_calc['predicted'], average='binary')
    print("\n\n> Recall Score =====>>>> ", rscore)

    ascore = accuracy_score(test_calc['y'], test_calc['predicted'])
    print("\n\n> Accuracy Score =====>>>> ", ascore)


if __name__ == '__main__':
    logistic_regression()

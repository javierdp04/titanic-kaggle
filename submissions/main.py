from src.main import build_main_model, DATA_DIR
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train_data = pd.read_csv(DATA_DIR / 'train.csv').copy()
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

    test_data = pd.read_csv(DATA_DIR / 'test.csv').copy()


    main_pipeline = build_main_model(n_estimators=500, max_depth=7, max_features=0.75, min_samples_leaf=1)
    main_pipeline.fit(X, y)

    submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": main_pipeline.predict(test_data),
    })
    submission.to_csv("submission.csv", index=False)


imagePath = 'C:\Users\emret\OneDrive\Masaüstü\proje_test\corn-leaf-8253_1280.jpg'; % Test görüntüsünün yolu
testImage = imread(imagePath); % Görüntüyü yükle
testImageResized = imresize(testImage, [224 224]); % Görüntüyü yeniden boyutlandır
predictedLabel = classify(proje_1, testImageResized); % Model tahmini

imshow(testImage); % Görüntüyü göster
title(['Tahmin: ', char(predictedLabel)]); % Tahmini başlık olarak yaz
disp(['Modelin Tahmini: ', char(predictedLabel)]);




import tensorflow as tf
from classification_models_3D.tfkeras import Classifiers
from tensorflow.keras.layers import Input, Flatten, ReLU, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def proxy_anchor_loss(embeddings, target, n_classes=2, input_dim=64, margin=0.1):
    '''
    ref: https://arxiv.org/abs/2003.13911
    official pytorch codes: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    '''
    alpha = 32
    delta = margin
    # define proxy weights
    proxy = tf.compat.v1.get_variable(
                        name='proxy',
                        shape=(n_classes, input_dim),
                        initializer=tf.random_normal_initializer(),
                        # initializer=tf.constant_initializer(1.0),
                        dtype=tf.float32,
                        trainable=True
                        )
    
    embeddings_l2 = tf.nn.l2_normalize(embeddings, axis=1)
    proxy_l2 = tf.nn.l2_normalize(proxy, axis=1)

    pos_target = tf.one_hot(target, n_classes, dtype=tf.float32)
    neg_target = 1.0 - pos_target

    sim_mat = tf.matmul(embeddings_l2, proxy_l2, transpose_b=True)

    pos_mat = tf.exp(-alpha * (sim_mat - delta)) * pos_target
    neg_mat = tf.exp(alpha * (sim_mat + delta)) * neg_target

    num_valid_proxies = n_classes
    pos_term = 1.0 / num_valid_proxies * tf.reduce_sum(tf.math.log(1.0 + tf.reduce_sum(pos_mat, axis=0)))
    neg_term = 1.0 / n_classes * tf.reduce_sum(tf.math.log(1.0 + tf.reduce_sum(neg_mat, axis=0)))

    loss = pos_term + neg_term

    return loss

def ProxyAnchorLoss(y_true, y_pred):
    return proxy_anchor_loss(y_pred, y_true)

def get_classification_models_network_3D_trip(
        lrVal=0.1,
        beta_1Val=0.999,
    ):
    IMG_SIZE = (40, 40, 40)
    embedding_size = 64
    # main model
    img_input_1 = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3))
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_networkR = ResNet18(
        input_tensor=img_input_1,
        weights='imagenet',
        include_top=False)
    x = base_networkR.layers[-1].output
    x = Flatten()(x)
    x = ReLU()(x)

    base_networkX = Dense(units=embedding_size,
        kernel_regularizer=regularizers.L1(1e-5),
        bias_regularizer=regularizers.L1(1e-4),
        activity_regularizer=regularizers.L1(1e-5)
    )(x)
    print("base_networkX = ", base_networkX)
    base_network = Model(img_input_1, base_networkX)
    base_network.summary()
    
    opt = Adam(lr=lrVal, beta_1=beta_1Val)  # choose optimiser. RMS is good too!

    base_network.compile(loss=ProxyAnchorLoss,
        optimizer=opt,
        run_eagerly=True
        )
    
    return base_network

def getStep2TrainModel(base_network, lrVal=0.1, beta_1Val=0.999):
    classifier_output = Dense(
        2,
        activation='softmax',
        kernel_regularizer=regularizers.L1(1e-5),
        bias_regularizer=regularizers.L1(1e-4),
        activity_regularizer=regularizers.L1(1e-5)

        )(base_network.output)

    class_model = Model(
        base_network.input,
        classifier_output,
        name='model')
    class_model.compile(
        optimizer=Adam(lr=lrVal, beta_1=beta_1Val),
        loss='binary_crossentropy',
        metrics=['accuracy']
        )
    
    
    for layer in class_model.layers[:-1]:
        layer.trainable=False

    return class_model
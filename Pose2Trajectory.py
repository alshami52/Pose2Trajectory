"""
Alsham's Code
Creates a Transformer-based model for time series prediction.

Args:
    src (int): The number of time steps in the input sequence.
    num_features (int): The number of features in the input sequence.
    ff_dim (int): The dimensionality of the feed-forward layer.
    
Returns:
    model (Model): The compiled Keras model.
"""
def create_model(src, num_features, ff_dim=512):
    # Encoder
    encoder_inputs = Input(shape=(src, num_features)) # Define input shape for the encoder
    lstm_encoder = LSTM(64, return_sequences=True, dropout=0.2)(encoder_inputs) # LSTM layer for the encoder
    time_embedding_1 = Time2Vector(src) # Time embedding layer for the encoder
    x = concat([lstm_encoder, time_embedding_1(lstm_encoder)], axis=-1) # Concatenate LSTM outputs and time features
    
    # Transformer encoder layers
    for _ in range(1):
        x = MultiHeadAttention(12, 512, 0.1)(x, x) # Multi-head self-attention
        x = LayerNormalization(epsilon=1e-6)(x) # Layer normalization
        x = TimeDistributed(Dense(ff_dim, activation='relu'))(x) # Feed-forward layer
        x = TimeDistributed(Dense(512))(x) # Feed-forward layer
        x = LayerNormalization(epsilon=1e-6)(x) # Layer normalization

    # Decoder
    decoder_inputs = Input(shape=(10, 2)) # Define input shape for the decoder
    lstm_decoder = LSTM(64, return_sequences=True, dropout=0.2)(decoder_inputs) # LSTM layer for the decoder
    time_embedding_2 = Time2Vector(10) # Time embedding layer for the decoder
    y = concat([lstm_decoder, time_embedding_2(lstm_decoder)], axis=-1) # Concatenate LSTM outputs and time features
    
    # Transformer decoder layers
    for _ in range(1):
        y = MultiHeadAttention(12, 512, 0.1)(y, y) # Multi-head self-attention
        y = MultiHeadAttention(12, 512, 0.1)(y, x) # Multi-head cross-attention
        y = LayerNormalization(epsilon=1e-6)(y) # Layer normalization
        y = TimeDistributed(Dense(ff_dim, activation="relu"))(y) # Feed-forward layer
        y = Dropout(0.1)(y) # Dropout layer
        y = Add()([y, y]) # Residual connection
        y = LayerNormalization(epsilon=1e-6)(y) # Layer normalization

    # Output layer
    decoder_outputs = TimeDistributed(Dense(2, activation="linear"))(y) # Linear output layer
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs) # Create the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mape']) # Compile the model

    return model

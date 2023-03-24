# Dependencies for Adversarial Machine Learning examples: https://github.com/sheatsley/attacks
FROM sheatsley/attacks
RUN git clone --depth 1 https://github.com/sheatsley/adversarial-attacks-pytorch.git \
    && git clone --depth 1 https://github.com/sheatsley/advertorch.git \
    && git clone --depth 1 https://github.com/sheatsley/adversarial-robustness-toolbox.git \
    && git clone --depth 1 https://github.com/sheatsley/cleverhans.git \
    && git clone --depth 1 https://github.com/sheatsley/foolbox.git \
    && pip install --no-cache-dir \
        adversarial-attacks-pytorch/ \
        adversarial-robustness-toolbox/ \
        advertorch/ \
        cleverhans/ \
        foolbox/ \
        matplotlib \
        plotly \
        seaborn \
    && rm -r *
WORKDIR /attacks/examples

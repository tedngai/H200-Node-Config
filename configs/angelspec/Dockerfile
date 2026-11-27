FROM lmsysorg/sglang@sha256:b872c66f40dee150709e5106d53f181719b1a4a4e8e950a908cfabff9be1a213

LABEL org.opencontainers.image.revision="82308b9d9e1685aa9404272cc499c4c195610913"
COPY requirements.txt /opt/angelspec/requirements.txt
# The two DeepSeek recipe distributions own the same import package.
RUN python3 -m pip uninstall -y deepseek-recipe \
    && python3 -m pip install --no-cache-dir --no-deps --require-hashes -r /opt/angelspec/requirements.txt

COPY python/sglang /opt/angelspec/python/sglang
COPY test/registered/unit/spec/test_dfly2_dsv41.py \
     test/registered/unit/managers/test_scheduler_sampling_mask_validation.py \
     /opt/angelspec/tests/
RUN printf '%s\n' '__version__ = "0.0.0.dev1+g82308b9d9"' \
    '__version_tuple__ = (0, 0, 0, "dev1", "g82308b9d9")' \
    > /opt/angelspec/python/sglang/_version.py

# Keep the nightly's CUDA libraries; do not load its older Rust cache bindings.
ENV PYTHONPATH=/opt/angelspec/python \
    SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python

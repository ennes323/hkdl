"""Command-family procedures and dedicated result presentation.

dispatch selects a handler after choosing common admission. Handlers translate
arguments into service calls and user reports; services own operation contracts.
Keep entrypoint and dispatch dependencies directed into this package.
"""

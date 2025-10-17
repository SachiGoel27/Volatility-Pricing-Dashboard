from django.db import models

class Underlying(models.Model):
    ticker = models.CharField(max_length=10)
    last_price = models.FloatField()
    timestamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.ticker


class OptionContract(models.Model):
    underlying = models.ForeignKey(Underlying, on_delete=models.CASCADE)
    expiration = models.DateField()
    strike = models.FloatField()
    option_type = models.CharField(max_length=4, choices=[('CALL', 'CALL'), ('PUT', 'PUT')])
    implied_vol = models.FloatField()
    last_price = models.FloatField()
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.underlying.ticker} {self.strike} {self.option_type}"

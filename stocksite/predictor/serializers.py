from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    date = serializers.DateField()
    actual = serializers.CharField()
    predicted = serializers.CharField()

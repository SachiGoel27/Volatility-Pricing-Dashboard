from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('page_b/', views.page_b_view, name='page_b'),
    path('page_c/', views.page_c_view, name='page_c'),
]